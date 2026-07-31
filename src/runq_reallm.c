/* Q8_0 inference for HETEROGENEOUS ReaLLM-Forge models in pure C.
 *
 * Int8 (Q8_0 group-quantized) variant of run_reallm.c: matmul weights are stored int8 with
 * per-group fp32 scales (norms stay fp32), roughly quartering model size for on-device use.
 * Reads version-2 `.rlm` files from reallmforge/export_reallm_hetero.py --version 2.
 * See doc/reallmforge_hetero_infinite.md. Same architecture support as run_reallm.c
 * (infinite/identity attention, per-layer dims, GQA, peri-LN, GeGLU + erf GELU, RoPE, tied cls).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include "bpe.h"

#define RLM_MAGIC 0x726C6D31u  // "rlm1"
#define ATTN_INFINITE 0
#define ATTN_IDENTITY 1

int GS = 0;  // group size for Q8_0 quantization (from header)

typedef struct {
    int8_t* q;   // quantized values
    float*  s;   // per-group scales
} QuantizedTensor;

typedef struct {
    int attn_type;
    int n_head, n_kv, qk, vd, mlp_hidden;
    int* head_to_kv;                 // [n_head] GQA group per head; NULL for identity
    float *pre_ln_attn, *peri_ln_attn, *pre_ln_mlp, *peri_ln_mlp;  // fp32 norms
    QuantizedTensor wq, wk, wv, wo;  // q8 (identity: .q == NULL)
    QuantizedTensor w1, w3, w2;      // q8 (gate, value, down)
} Layer;

typedef struct {
    int n_layer, n_embd, vocab_size, seq_len, shared, act;
    float rope_theta;
} Config;

typedef struct {
    Layer* layers;
    float* ln_f;                     // final RMSNorm gain
    QuantizedTensor q_tokens;        // quantized token embedding (== classifier if shared)
    float* token_embedding_table;    // dequantized wte, for the embedding lookup
    QuantizedTensor wcls;            // classifier
} Weights;

typedef struct {
    float *x, *xb, *xb2;
    QuantizedTensor xq;              // quantized activation scratch (sized to max matmul in-dim)
    float *q, *att, *headcat, *hb, *hb2, *logits;
    float **key_cache, **value_cache;
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed (%zu bytes)\n", n); exit(EXIT_FAILURE); }
    return p;
}
static int imax(int a, int b) { return a > b ? a : b; }

// ---- Q8_0 primitives (identical to runq.c) ----
void dequantize(QuantizedTensor* qx, float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = qx->q[i] * qx->s[i / GS];
}
void quantize(QuantizedTensor* qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        if (scale == 0.0f) scale = 1.0f;   // guard all-zero group (avoids 0/0 NaN)
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t) roundf(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}
void matmul(float* xout, QuantizedTensor* x, QuantizedTensor* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,), both quantized; requires n % GS == 0
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++)
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

// map one Q8_0 tensor of `numel` elements from *p, advancing *p past int8 values + fp32 scales
static QuantizedTensor map_q8(char** p, long numel) {
    QuantizedTensor qt;
    qt.q = (int8_t*)(*p); *p += numel;
    qt.s = (float*)(*p);  *p += (numel / GS) * sizeof(float);
    return qt;
}

static int* compute_head_to_kv(int n_head, int n_kv) {
    int* map = xmalloc(n_head * sizeof(int));
    int base = n_head / n_kv, rem = n_head % n_kv, idx = 0;
    for (int g = 0; g < n_kv; g++) {
        int gs = base + (g < rem ? 1 : 0);
        for (int j = 0; j < gs; j++) map[idx++] = g;
    }
    if (idx != n_head) { fprintf(stderr, "bad kv-group distribution n_head=%d n_kv=%d\n", n_head, n_kv); exit(EXIT_FAILURE); }
    return map;
}

void memory_map_weights(Transformer* t) {
    Config* c = &t->config;
    Weights* w = &t->weights;
    char* base = (char*)t->data;
    int* desc = (int*)(base + 256);
    char* p = base + 256 + (size_t)c->n_layer * 8 * sizeof(int);
    int dim = c->n_embd;

    w->layers = xmalloc(c->n_layer * sizeof(Layer));
    for (int l = 0; l < c->n_layer; l++) {
        Layer* L = &w->layers[l];
        L->attn_type  = desc[l*8 + 0];
        L->n_head     = desc[l*8 + 1];
        L->n_kv       = desc[l*8 + 2];
        L->qk         = desc[l*8 + 3];
        L->vd         = desc[l*8 + 4];
        L->mlp_hidden = desc[l*8 + 5];
        L->head_to_kv = NULL;

        L->pre_ln_attn  = (float*)p; p += (size_t)dim * sizeof(float);
        L->peri_ln_attn = (float*)p; p += (size_t)dim * sizeof(float);
        if (L->attn_type == ATTN_INFINITE) {
            L->head_to_kv = compute_head_to_kv(L->n_head, L->n_kv);
            L->wq = map_q8(&p, (long)L->n_head * L->qk * dim);
            L->wk = map_q8(&p, (long)L->n_kv   * L->qk * dim);
            L->wv = map_q8(&p, (long)L->n_kv   * L->vd * dim);
            L->wo = map_q8(&p, (long)dim * (L->n_head * L->vd));
        } else {
            L->wq.q = L->wk.q = L->wv.q = L->wo.q = NULL;
        }
        L->pre_ln_mlp  = (float*)p; p += (size_t)dim * sizeof(float);
        L->peri_ln_mlp = (float*)p; p += (size_t)dim * sizeof(float);
        L->w1 = map_q8(&p, (long)L->mlp_hidden * dim);
        L->w3 = map_q8(&p, (long)L->mlp_hidden * dim);
        L->w2 = map_q8(&p, (long)dim * L->mlp_hidden);
    }
    w->ln_f = (float*)p; p += (size_t)dim * sizeof(float);
    w->q_tokens = map_q8(&p, (long)c->vocab_size * dim);
    w->token_embedding_table = xmalloc((size_t)c->vocab_size * dim * sizeof(float));
    dequantize(&w->q_tokens, w->token_embedding_table, c->vocab_size * dim);
    w->wcls = c->shared ? w->q_tokens : map_q8(&p, (long)c->vocab_size * dim);
}

void malloc_run_state(RunState* s, Config* p, Weights* w) {
    int dim = p->n_embd;
    int max_qdim = 1, max_att = 1, max_headcat = 1, max_hidden = 1;
    for (int l = 0; l < p->n_layer; l++) {
        Layer* L = &w->layers[l];
        max_hidden = imax(max_hidden, L->mlp_hidden);
        if (L->attn_type == ATTN_INFINITE) {
            max_qdim    = imax(max_qdim, L->n_head * L->qk);
            max_att     = imax(max_att, L->n_head * p->seq_len);
            max_headcat = imax(max_headcat, L->n_head * L->vd);
        }
    }
    int max_in = imax(dim, imax(max_headcat, max_hidden));  // largest quantized matmul in-dim
    s->x       = xmalloc(dim * sizeof(float));
    s->xb      = xmalloc(dim * sizeof(float));
    s->xb2     = xmalloc(dim * sizeof(float));
    s->xq.q    = xmalloc(max_in * sizeof(int8_t));
    s->xq.s    = xmalloc((max_in / GS) * sizeof(float));
    s->q       = xmalloc(max_qdim * sizeof(float));
    s->att     = xmalloc(max_att * sizeof(float));
    s->headcat = xmalloc(max_headcat * sizeof(float));
    s->hb      = xmalloc(max_hidden * sizeof(float));
    s->hb2     = xmalloc(max_hidden * sizeof(float));
    s->logits  = xmalloc(p->vocab_size * sizeof(float));
    s->key_cache   = xmalloc(p->n_layer * sizeof(float*));
    s->value_cache = xmalloc(p->n_layer * sizeof(float*));
    for (int l = 0; l < p->n_layer; l++) {
        Layer* L = &w->layers[l];
        if (L->attn_type == ATTN_INFINITE) {
            s->key_cache[l]   = xmalloc((size_t)p->seq_len * L->n_kv * L->qk * sizeof(float));
            s->value_cache[l] = xmalloc((size_t)p->seq_len * L->n_kv * L->vd * sizeof(float));
        } else {
            s->key_cache[l] = NULL; s->value_cache[l] = NULL;
        }
    }
}

void free_run_state(RunState* s, Config* p) {
    free(s->x); free(s->xb); free(s->xb2); free(s->xq.q); free(s->xq.s);
    free(s->q); free(s->att); free(s->headcat); free(s->hb); free(s->hb2); free(s->logits);
    for (int l = 0; l < p->n_layer; l++) { free(s->key_cache[l]); free(s->value_cache[l]); }
    free(s->key_cache); free(s->value_cache);
}

void read_checkpoint(char* path, Transformer* t) {
    FILE* file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "couldn't open %s\n", path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    t->file_size = ftell(file);
    fclose(file);

    t->fd = open(path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    uint32_t* hu = (uint32_t*)t->data;
    int* hi = (int*)t->data;
    float* hf = (float*)t->data;
    if (hu[0] != RLM_MAGIC) { fprintf(stderr, "bad magic 0x%08x (not a .rlm file)\n", hu[0]); exit(EXIT_FAILURE); }
    int version = hi[1];
    if (version != 2) { fprintf(stderr, "runq_reallm needs a version-2 (Q8_0) .rlm; got version %d (use run_reallm for fp32)\n", version); exit(EXIT_FAILURE); }
    Config* c = &t->config;
    c->n_layer    = hi[2];
    c->n_embd     = hi[3];
    c->vocab_size = hi[4];
    c->seq_len    = hi[5];
    c->shared     = hi[6];
    c->act        = hi[7];
    c->rope_theta = hf[8];
    GS            = hi[9];
    if (GS <= 0) { fprintf(stderr, "bad group_size %d\n", GS); exit(EXIT_FAILURE); }

    memory_map_weights(t);
}

void build_transformer(Transformer* t, char* path) {
    read_checkpoint(path, t);
    malloc_run_state(&t->state, &t->config, &t->weights);
}

void free_transformer(Transformer* t) {
    for (int l = 0; l < t->config.n_layer; l++) free(t->weights.layers[l].head_to_kv);
    free(t->weights.layers);
    free(t->weights.token_embedding_table);
    if (t->data != MAP_FAILED && t->data != NULL) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
    free_run_state(&t->state, &t->config);
}

// ---- neural net blocks ----
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss = 1.0f / sqrtf(ss);          // NO eps (matches ReaLLM-Forge RMSNorm)
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}
static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}
static void rope_apply(float* vec, int n_groups, int head_size, int pos, float theta) {
    for (int h = 0; h < n_groups; h++) {
        float* v = vec + (size_t)h * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)head_size);
            float angle = pos * freq;
            float fcr = cosf(angle), fci = sinf(angle);
            float v0 = v[i], v1 = v[i + 1];
            v[i]     = v0 * fcr - v1 * fci;
            v[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

float* forward(Transformer* t, int token, int pos) {
    Config* c = &t->config;
    Weights* w = &t->weights;
    RunState* s = &t->state;
    int dim = c->n_embd;
    float* x = s->x;

    memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layer; l++) {
        Layer* L = &w->layers[l];

        // ---- attention sub-block ----
        rmsnorm(s->xb, x, L->pre_ln_attn, dim);
        if (L->attn_type == ATTN_IDENTITY) {
            rmsnorm(s->xb2, s->xb, L->peri_ln_attn, dim);
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        } else {
            int n_head = L->n_head, n_kv = L->n_kv, qk = L->qk, vd = L->vd;
            int kvk = n_kv * qk, kvv = n_kv * vd;
            float* q = s->q;
            float* kdst = s->key_cache[l]   + (size_t)pos * kvk;
            float* vdst = s->value_cache[l] + (size_t)pos * kvv;
            quantize(&s->xq, s->xb, dim);
            matmul(q,    &s->xq, &L->wq, dim, n_head * qk);
            matmul(kdst, &s->xq, &L->wk, dim, kvk);
            matmul(vdst, &s->xq, &L->wv, dim, kvv);
            rope_apply(q,    n_head, qk, pos, c->rope_theta);
            rope_apply(kdst, n_kv,   qk, pos, c->rope_theta);

            float scale = 1.0f / sqrtf((float)qk);
            int h;
            #pragma omp parallel for private(h)
            for (h = 0; h < n_head; h++) {
                int g = L->head_to_kv[h];
                float* qh = q + (size_t)h * qk;
                float* att = s->att + (size_t)h * c->seq_len;
                for (int tt = 0; tt <= pos; tt++) {
                    float* kt = s->key_cache[l] + (size_t)tt * kvk + (size_t)g * qk;
                    float sc = 0.0f;
                    for (int i = 0; i < qk; i++) sc += qh[i] * kt[i];
                    att[tt] = sc * scale;
                }
                softmax(att, pos + 1);
                float* outh = s->headcat + (size_t)h * vd;
                for (int i = 0; i < vd; i++) outh[i] = 0.0f;
                for (int tt = 0; tt <= pos; tt++) {
                    float a = att[tt];
                    float* vt = s->value_cache[l] + (size_t)tt * kvv + (size_t)g * vd;
                    for (int i = 0; i < vd; i++) outh[i] += a * vt[i];
                }
            }
            quantize(&s->xq, s->headcat, n_head * vd);
            matmul(s->xb, &s->xq, &L->wo, n_head * vd, dim);
            rmsnorm(s->xb2, s->xb, L->peri_ln_attn, dim);
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        }

        // ---- MLP sub-block (GeGLU) ----
        rmsnorm(s->xb, x, L->pre_ln_mlp, dim);
        int hidden = L->mlp_hidden;
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb,  &s->xq, &L->w1, dim, hidden);
        matmul(s->hb2, &s->xq, &L->w3, dim, hidden);
        for (int i = 0; i < hidden; i++) s->hb[i] = gelu_erf(s->hb[i]) * s->hb2[i];
        quantize(&s->xq, s->hb, hidden);
        matmul(s->xb, &s->xq, &L->w2, hidden, dim);
        rmsnorm(s->xb2, s->xb, L->peri_ln_mlp, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
    }

    rmsnorm(x, x, w->ln_f, dim);
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, &w->wcls, dim, c->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// sampler (verbatim from run.c)

typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size; ProbIndex* probindex;
    float temperature; float topp; unsigned long long rng_state;
} Sampler;

int sample_argmax(float* p, int n) {
    int mi = 0; float mp = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mp) { mi = i; mp = p[i]; }
    return mi;
}
int sample_mult(float* p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a; ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}
int sample_topp(float* p, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (p[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = p[i]; n0++; }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cum = 0.0f; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += probindex[i].prob; if (cum > topp) { last = i; break; } }
    float r = coin * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; }
    return probindex[last].index;
}
void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size; s->temperature = temperature; s->topp = topp; s->rng_state = seed;
    s->probindex = xmalloc(vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler* s) { free(s->probindex); }
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample(Sampler* s, float* logits) {
    int next;
    if (s->temperature == 0.0f) {
        next = sample_argmax(logits, s->vocab_size);
    } else {
        for (int q = 0; q < s->vocab_size; q++) logits[q] /= s->temperature;
        softmax(logits, s->vocab_size);
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0 || s->topp >= 1) next = sample_mult(logits, s->vocab_size, coin);
        else next = sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
    }
    return next;
}

long time_in_ms() {
    struct timespec time; clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation

void generate_ids(Transformer* t, Sampler* sampler, char* ids_path, int steps) {
    FILE* f = fopen(ids_path, "r");
    if (!f) { fprintf(stderr, "couldn't open ids file %s\n", ids_path); exit(EXIT_FAILURE); }
    int cap = 256, n = 0;
    int* prompt = xmalloc(cap * sizeof(int));
    int tok;
    while (fscanf(f, "%d", &tok) == 1) {
        if (n == cap) { cap *= 2; prompt = realloc(prompt, cap * sizeof(int)); }
        prompt[n++] = tok;
    }
    fclose(f);
    if (n < 1) { fprintf(stderr, "no token ids read from %s\n", ids_path); exit(EXIT_FAILURE); }
    int vsz = t->config.vocab_size;
    for (int i = 0; i < n; i++) {
        if (prompt[i] < 0 || prompt[i] >= vsz) {
            fprintf(stderr, "token id %d at %d out of range [0,%d)\n", prompt[i], i, vsz);
            exit(EXIT_FAILURE);
        }
    }
    int next, token = prompt[0], pos = 0;
    while (pos < steps) {
        float* logits = forward(t, token, pos);
        if (pos < n - 1) next = prompt[pos + 1];
        else next = sample(sampler, logits);
        pos++;
        printf("%d\n", next);
        token = next;
    }
    fflush(stdout);
    free(prompt);
}

void generate_gpt2(Transformer* t, Sampler* sampler, GPT2Tokenizer* tok, char* prompt, int steps) {
    if (prompt == NULL) prompt = "";
    int* prompt_tokens = xmalloc((strlen(prompt) + 1) * sizeof(int));
    int num_prompt_tokens = 0;
    gpt2_encode(tok, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { prompt_tokens[0] = tok->eot; num_prompt_tokens = 1; }

    { int blen0; const unsigned char* b0 = gpt2_token_bytes(tok, prompt_tokens[0], &blen0);
      fwrite(b0, 1, blen0, stdout); }

    long start = 0;
    int next, token = prompt_tokens[0], pos = 0;
    while (pos < steps) {
        float* logits = forward(t, token, pos);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1];
        else next = sample(sampler, logits);
        pos++;
        if (next == tok->eot) break;
        int blen; const unsigned char* b = gpt2_token_bytes(tok, next, &blen);
        fwrite(b, 1, blen, stdout); fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq_reallm <model.q8.rlm> [options]\n");
    fprintf(stderr, "Example: runq_reallm model.q8.rlm -g tokenizer_gpt2.bin -i \"Once upon a time\" -t 0.8\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed\n");
    fprintf(stderr, "  -n <int>    number of steps, default 256\n");
    fprintf(stderr, "  -i <string> input prompt (with -g)\n");
    fprintf(stderr, "  -g <string> GPT-2 BPE tokenizer .bin\n");
    fprintf(stderr, "  -I <string> token-ID prompt file: raw IDs in, raw IDs out (parity check)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    char* checkpoint_path = NULL;
    float temperature = 1.0f, topp = 0.9f;
    int steps = 256;
    char* prompt = NULL;
    unsigned long long rng_seed = 0;
    char* ids_path = NULL;
    char* gpt2_path = NULL;

    if (argc >= 2) checkpoint_path = argv[1]; else error_usage();
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        if (argv[i][0] != '-') error_usage();
        if (strlen(argv[i]) != 2) error_usage();
        switch (argv[i][1]) {
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = atoi(argv[i + 1]); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'I': ids_path = argv[i + 1]; break;
            case 'g': gpt2_path = argv[i + 1]; break;
            default: error_usage();
        }
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (ids_path != NULL) {
        generate_ids(&transformer, &sampler, ids_path, steps);
    } else if (gpt2_path != NULL) {
        GPT2Tokenizer gtok = gpt2_load(gpt2_path);
        generate_gpt2(&transformer, &sampler, &gtok, prompt, steps);
        gpt2_free(&gtok);
    } else {
        fprintf(stderr, "this runner needs -g <tokenizer_gpt2.bin> (text) or -I <ids_file> (parity)\n");
        error_usage();
    }

    free_sampler(&sampler);
    free_transformer(&transformer);
    return 0;
}
