/* Inference for HETEROGENEOUS ReaLLM-Forge models in pure C.
 *
 * Runs models with per-layer dims and "infinite head" attention (concat path),
 * identity attention layers, GQA, peri-LN, GeGLU MLP (exact erf GELU), RoPE, and
 * a tied classifier. Reads the self-describing `.rlm` format produced by
 * reallmforge/export_reallm_hetero.py. See doc/reallmforge_hetero_infinite.md.
 *
 * fp32 only (parity focus). Reuses the llama2.c sampler / CLI / bpe.h tokenizer.
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
#include "bpe.h"  // GPT-2 byte-level BPE tokenizer (tiktoken-gpt2 models)

// ----------------------------------------------------------------------------
// model

#define RLM_MAGIC 0x726C6D31u  // "rlm1"
#define ATTN_INFINITE 0
#define ATTN_IDENTITY 1

typedef struct {
    int attn_type;                 // ATTN_INFINITE | ATTN_IDENTITY
    int n_head, n_kv, qk, vd, mlp_hidden;
    int* head_to_kv;               // [n_head] group index per head (GQA), NULL for identity
    // weight pointers into the mmap'd file
    float *pre_ln_attn, *peri_ln_attn;
    float *wq, *wk, *wv, *wo;      // NULL for identity
    float *pre_ln_mlp, *peri_ln_mlp;
    float *w1, *w3, *w2;           // gate, value, down
} Layer;

typedef struct {
    int n_layer, n_embd, vocab_size, seq_len, shared, act;
    float rope_theta;
} Config;

typedef struct {
    Layer* layers;
    float* ln_f;    // final RMSNorm gain (n_embd)
    float* wte;     // token embedding (vocab, n_embd)
    float* wcls;    // classifier (== wte if shared)
} Weights;

typedef struct {
    float *x, *xb, *xb2;   // residual stream + scratch (n_embd)
    float *q;              // query buffer (max n_head*qk)
    float *att;            // attention scores (max n_head * seq_len)
    float *headcat;        // concatenated head outputs (max n_head*vd)
    float *hb, *hb2;       // mlp hidden buffers (max mlp_hidden)
    float *logits;         // output logits (vocab)
    float **key_cache;     // [n_layer] each seq_len*(n_kv*qk); NULL for identity
    float **value_cache;   // [n_layer] each seq_len*(n_kv*vd); NULL for identity
} RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState state;
    int fd;
    float* data;           // mmap base
    ssize_t file_size;
} Transformer;

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "malloc failed (%zu bytes)\n", n); exit(EXIT_FAILURE); }
    return p;
}

static int imax(int a, int b) { return a > b ? a : b; }

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
    s->x       = xmalloc(dim * sizeof(float));
    s->xb      = xmalloc(dim * sizeof(float));
    s->xb2     = xmalloc(dim * sizeof(float));
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
            s->key_cache[l] = NULL;
            s->value_cache[l] = NULL;
        }
    }
}

void free_run_state(RunState* s, Config* p) {
    free(s->x); free(s->xb); free(s->xb2); free(s->q); free(s->att);
    free(s->headcat); free(s->hb); free(s->hb2); free(s->logits);
    for (int l = 0; l < p->n_layer; l++) { free(s->key_cache[l]); free(s->value_cache[l]); }
    free(s->key_cache); free(s->value_cache);
}

// _compute_kv_group_distribution: contiguous groups, first (n_head%n_kv) get one extra
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

// walk the mmap'd weight region and wire up all pointers
void memory_map_weights(Transformer* t) {
    Config* c = &t->config;
    Weights* w = &t->weights;
    char* base = (char*)t->data;
    int* desc = (int*)(base + 256);            // per-layer descriptor table
    float* p = (float*)(base + 256 + (size_t)c->n_layer * 8 * sizeof(int));  // weights start
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

        L->pre_ln_attn  = p; p += dim;
        L->peri_ln_attn = p; p += dim;
        if (L->attn_type == ATTN_INFINITE) {
            L->head_to_kv = compute_head_to_kv(L->n_head, L->n_kv);
            L->wq = p; p += (size_t)L->n_head * L->qk * dim;
            L->wk = p; p += (size_t)L->n_kv   * L->qk * dim;
            L->wv = p; p += (size_t)L->n_kv   * L->vd * dim;
            L->wo = p; p += (size_t)dim * (L->n_head * L->vd);
        } else {
            L->wq = L->wk = L->wv = L->wo = NULL;
        }
        L->pre_ln_mlp  = p; p += dim;
        L->peri_ln_mlp = p; p += dim;
        L->w1 = p; p += (size_t)L->mlp_hidden * dim;   // gate
        L->w3 = p; p += (size_t)L->mlp_hidden * dim;   // value
        L->w2 = p; p += (size_t)dim * L->mlp_hidden;   // down
    }
    w->ln_f = p; p += dim;
    w->wte = p; p += (size_t)c->vocab_size * dim;
    if (c->shared) {
        w->wcls = w->wte;
    } else {
        w->wcls = p; p += (size_t)c->vocab_size * dim;
    }
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
    if (version != 1) { fprintf(stderr, "unsupported .rlm version %d\n", version); exit(EXIT_FAILURE); }
    Config* c = &t->config;
    c->n_layer    = hi[2];
    c->n_embd     = hi[3];
    c->vocab_size = hi[4];
    c->seq_len    = hi[5];
    c->shared     = hi[6];
    c->act        = hi[7];
    c->rope_theta = hf[8];

    memory_map_weights(t);
}

void build_transformer(Transformer* t, char* path) {
    read_checkpoint(path, t);
    malloc_run_state(&t->state, &t->config, &t->weights);
}

void free_transformer(Transformer* t) {
    for (int l = 0; l < t->config.n_layer; l++) free(t->weights.layers[l].head_to_kv);
    free(t->weights.layers);
    if (t->data != MAP_FAILED && t->data != NULL) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
    free_run_state(&t->state, &t->config);
}

// ----------------------------------------------------------------------------
// neural net blocks

// RMSNorm WITHOUT epsilon, matching ReaLLM-Forge: rms = ||x||/sqrt(d); y = x/rms * gain
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;                    // mean square
    ss = 1.0f / sqrtf(ss);         // 1/rms  (NO eps)
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
        xout[i] = val;
    }
}

static inline float gelu_erf(float x) {
    // exact erf GELU (matches nn.GELU() default approximate='none')
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// interleaved RoPE over each head's qk dim; vec is (n_groups * head_size) contiguous
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

    memcpy(x, w->wte + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layer; l++) {
        Layer* L = &w->layers[l];

        // ---- attention sub-block ----
        rmsnorm(s->xb, x, L->pre_ln_attn, dim);        // x_attn_in
        if (L->attn_type == ATTN_IDENTITY) {
            // attn is a pass-through: attn_out = x_attn_in
            rmsnorm(s->xb2, s->xb, L->peri_ln_attn, dim);
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        } else {
            int n_head = L->n_head, n_kv = L->n_kv, qk = L->qk, vd = L->vd;
            int kvk = n_kv * qk, kvv = n_kv * vd;
            float* q = s->q;
            float* kdst = s->key_cache[l]   + (size_t)pos * kvk;
            float* vdst = s->value_cache[l] + (size_t)pos * kvv;
            matmul(q,    s->xb, L->wq, dim, n_head * qk);
            matmul(kdst, s->xb, L->wk, dim, kvk);
            matmul(vdst, s->xb, L->wv, dim, kvv);
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
            // concat heads (n_head*vd) -> c_proj -> attn_out
            matmul(s->xb, s->headcat, L->wo, n_head * vd, dim);
            rmsnorm(s->xb2, s->xb, L->peri_ln_attn, dim);
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
        }

        // ---- MLP sub-block (GeGLU) ----
        rmsnorm(s->xb, x, L->pre_ln_mlp, dim);
        int hidden = L->mlp_hidden;
        matmul(s->hb,  s->xb, L->w1, dim, hidden);     // gate
        matmul(s->hb2, s->xb, L->w3, dim, hidden);     // value
        for (int i = 0; i < hidden; i++) s->hb[i] = gelu_erf(s->hb[i]) * s->hb2[i];
        matmul(s->xb, s->hb, L->w2, hidden, dim);      // down -> mlp_out
        rmsnorm(s->xb2, s->xb, L->peri_ln_mlp, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
    }

    rmsnorm(x, x, w->ln_f, dim);
    matmul(s->logits, x, w->wcls, dim, c->vocab_size);
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

// validation path: token-ID prompt in, raw token IDs out (mirror run.c generate_ids)
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

// GPT-2 BPE generation (no BOS; stop on <|endoftext|>)
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
    fprintf(stderr, "Usage:   run_reallm <model.rlm> [options]\n");
    fprintf(stderr, "Example: run_reallm model.rlm -g tokenizer_gpt2.bin -i \"Once upon a time\" -t 0.8\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  top-p in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed\n");
    fprintf(stderr, "  -n <int>    number of steps, default 256\n");
    fprintf(stderr, "  -i <string> input prompt (with -g)\n");
    fprintf(stderr, "  -g <string> GPT-2 BPE tokenizer .bin (tiktoken-gpt2 models)\n");
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
