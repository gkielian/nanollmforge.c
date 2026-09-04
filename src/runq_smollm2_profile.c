/* Profiling Inference for SmolLM2-135M (Tier-1 Transformer) in pure C with OpenMP 4-Core Support.
 *
 * Measures fine-grained breakdown of:
 * 1. Embedding lookup
 * 2. Attention (QKV + RoPE + Softmax + Proj across 30 layers)
 * 3. MLP / FeedForward (Gate + Value + GeGLU + Down across 30 layers)
 * 4. LM Head (Final RMSNorm + Classifier Matmul 576 -> 50,257)
 * 5. Sampling (Logits argmax / top-p)
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

int GS = 0;

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    int8_t* q;
    float* s;
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    float* rms_final_weight;
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2;
    QuantizedTensor xq, hq;
    float *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

typedef struct {
    double us_embed;
    double us_attn_norm;
    double us_attn_qkv;
    double us_attn_rope;
    double us_attn_core;
    double us_attn_proj;
    double us_mlp_norm;
    double us_mlp_gate_val;
    double us_mlp_act;
    double us_mlp_down;
    double us_lm_head_norm;
    double us_lm_head_matmul;
    double us_sample;
    int count;
} StepProfile;

static inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = (int8_t*)calloc(p->dim, sizeof(int8_t)), .s = (float*)calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = (int8_t*)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float*)calloc(p->hidden_dim, sizeof(float)) };
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc_run_state failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->xq.q); free(s->xq.s); free(s->hq.q); free(s->hq.s);
    free(s->q); free(s->k); free(s->v); free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
}

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabsf(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = scale > 0.0f ? (x[group * GS + i] / scale) : 0.0f;
            qx->q[group * GS + i] = (int8_t)roundf(quant_value);
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, size_t size_each) {
    void *p = *ptr;
    QuantizedTensor *res = (QuantizedTensor*)malloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        res[i].s = (float*)p;
        p = (float*)p + (size_each / GS);
    }
    *ptr = p;
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    float* fptr = (float*) ptr;
    w->rms_att_weight = fptr;
    fptr += (size_t)p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += (size_t)p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, (size_t)p->vocab_size * p->dim);
    w->token_embedding_table = (float*)malloc((size_t)p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (size_t)(p->n_heads * head_size) * p->dim);
    w->w1 = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, (size_t)p->dim * p->hidden_dim);
    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, (size_t)p->dim * p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 || magic_number != 0x616b3432) {
        fprintf(stderr, "Bad magic number in checkpoint (expected ak42)\n"); exit(EXIT_FAILURE);
    }
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1 || version != 2) {
        fprintf(stderr, "Bad version %d in checkpoint (expected 2)\n", version); exit(EXIT_FAILURE);
    }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size;
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size;
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + 256;
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void free_transformer(Transformer* t) {
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv); free(t->weights.wo);
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) { x[i] *= inv_sum; }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int in = i * n;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (GS == 16) {
            int num_groups = n / 16;
            for (int g = 0; g < num_groups; g++) {
                int j = g * 16;
                int8x16_t vx = vld1q_s8(&x->q[j]);
                int8x16_t vw = vld1q_s8(&w->q[in + j]);
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t dot = vdotq_s32(vdupq_n_s32(0), vx, vw);
                int32x2_t r = vadd_s32(vget_low_s32(dot), vget_high_s32(dot));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(vx), vget_low_s8(vw));
                int16x8_t p1 = vmull_s8(vget_high_s8(vx), vget_high_s8(vw));
                int32x4_t s0 = vpaddlq_s16(p0);
                int32x4_t s1 = vpadalq_s16(s0, p1);
                int32x2_t r = vadd_s32(vget_low_s32(s1), vget_high_s32(s1));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#endif
                val += ((float) ival) * w->s[(in + j) / 16] * x->s[g];
            }
        } else if (GS == 32) {
            int num_groups = n / 32;
            for (int g = 0; g < num_groups; g++) {
                int j = g * 32;
                int8x16_t vx0 = vld1q_s8(&x->q[j]);
                int8x16_t vw0 = vld1q_s8(&w->q[in + j]);
                int8x16_t vx1 = vld1q_s8(&x->q[j + 16]);
                int8x16_t vw1 = vld1q_s8(&w->q[in + j + 16]);
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t dot0 = vdotq_s32(vdupq_n_s32(0), vx0, vw0);
                int32x4_t dot1 = vdotq_s32(dot0, vx1, vw1);
                int32x2_t r = vadd_s32(vget_low_s32(dot1), vget_high_s32(dot1));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(vx0), vget_low_s8(vw0));
                int16x8_t p1 = vmull_s8(vget_high_s8(vx0), vget_high_s8(vw0));
                int16x8_t p2 = vmull_s8(vget_low_s8(vx1), vget_low_s8(vw1));
                int16x8_t p3 = vmull_s8(vget_high_s8(vx1), vget_high_s8(vw1));
                int32x4_t s0 = vpaddlq_s16(p0);
                int32x4_t s1 = vpadalq_s16(s0, p1);
                int32x4_t s2 = vpadalq_s16(s1, p2);
                int32x4_t s3 = vpadalq_s16(s2, p3);
                int32x2_t r = vadd_s32(vget_low_s32(s3), vget_high_s32(s3));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#endif
                val += ((float) ival) * w->s[(in + j) / 32] * x->s[g];
            }
        } else if (GS == 64) {
            int num_groups = n / 64;
            for (int g = 0; g < num_groups; g++) {
                int j = g * 64;
                int8x16_t vx0 = vld1q_s8(&x->q[j]);
                int8x16_t vw0 = vld1q_s8(&w->q[in + j]);
                int8x16_t vx1 = vld1q_s8(&x->q[j + 16]);
                int8x16_t vw1 = vld1q_s8(&w->q[in + j + 16]);
                int8x16_t vx2 = vld1q_s8(&x->q[j + 32]);
                int8x16_t vw2 = vld1q_s8(&w->q[in + j + 32]);
                int8x16_t vx3 = vld1q_s8(&x->q[j + 48]);
                int8x16_t vw3 = vld1q_s8(&w->q[in + j + 48]);
#if defined(__ARM_FEATURE_DOTPROD)
                int32x4_t dot0 = vdotq_s32(vdupq_n_s32(0), vx0, vw0);
                int32x4_t dot1 = vdotq_s32(dot0, vx1, vw1);
                int32x4_t dot2 = vdotq_s32(dot1, vx2, vw2);
                int32x4_t dot3 = vdotq_s32(dot2, vx3, vw3);
                int32x2_t r = vadd_s32(vget_low_s32(dot3), vget_high_s32(dot3));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(vx0), vget_low_s8(vw0));
                int16x8_t p1 = vmull_s8(vget_high_s8(vx0), vget_high_s8(vw0));
                int16x8_t p2 = vmull_s8(vget_low_s8(vx1), vget_low_s8(vw1));
                int16x8_t p3 = vmull_s8(vget_high_s8(vx1), vget_high_s8(vw1));
                int32x4_t s0 = vpaddlq_s16(p0);
                int32x4_t s1 = vpadalq_s16(s0, p1);
                int32x4_t s2 = vpadalq_s16(s1, p2);
                int32x4_t s3 = vpadalq_s16(s2, p3);
                int16x8_t p4 = vmull_s8(vget_low_s8(vx2), vget_low_s8(vw2));
                int16x8_t p5 = vmull_s8(vget_high_s8(vx2), vget_high_s8(vw2));
                int16x8_t p6 = vmull_s8(vget_low_s8(vx3), vget_low_s8(vw3));
                int16x8_t p7 = vmull_s8(vget_high_s8(vx3), vget_high_s8(vw3));
                int32x4_t s4 = vpadalq_s16(s3, p4);
                int32x4_t s5 = vpadalq_s16(s4, p5);
                int32x4_t s6 = vpadalq_s16(s5, p6);
                int32x4_t s7 = vpadalq_s16(s6, p7);
                int32x2_t r = vadd_s32(vget_low_s32(s7), vget_high_s32(s7));
                int32_t ival = vget_lane_s32(r, 0) + vget_lane_s32(r, 1);
#endif
                val += ((float) ival) * w->s[(in + j) / 64] * x->s[g];
            }
        } else {
            for (int j = 0; j < n; j++) {
                val += ((float) x->q[j]) * ((float) w->q[in + j]) * w->s[(in + j) / GS] * x->s[j / GS];
            }
        }
#else
        for (int j = 0; j < n; j++) {
            val += ((float) x->q[j]) * ((float) w->q[in + j]) * w->s[(in + j) / GS] * x->s[j / GS];
        }
#endif
        xout[i] = val;
    }
}

float* forward_profiled(Transformer* transformer, int token, int pos, int compute_logits, StepProfile *prof) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    double t0 = get_time_us();
    memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));
    double t1 = get_time_us();
    prof->us_embed += (t1 - t0);

    for (int l = 0; l < p->n_layers; l++) {
        double ta0 = get_time_us();
        rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);
        double ta1 = get_time_us();
        prof->us_attn_norm += (ta1 - ta0);

        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
        double ta2 = get_time_us();
        prof->us_attn_qkv += (ta2 - ta1);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        double ta3 = get_time_us();
        prof->us_attn_rope += (ta3 - ta2);

        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + (size_t)h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf((float)head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        double ta4 = get_time_us();
        prof->us_attn_core += (ta4 - ta3);

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }
        double ta5 = get_time_us();
        prof->us_attn_proj += (ta5 - ta4);

        // MLP Profiling
        double tm0 = get_time_us();
        rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);
        double tm1 = get_time_us();
        prof->us_mlp_norm += (tm1 - tm0);

        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        double tm2 = get_time_us();
        prof->us_mlp_gate_val += (tm2 - tm1);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val = 0.5f * val * (1.0f + erff(val * 0.70710678118654752440f));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        double tm3 = get_time_us();
        prof->us_mlp_act += (tm3 - tm2);

        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
        double tm4 = get_time_us();
        prof->us_mlp_down += (tm4 - tm3);
    }

    if (!compute_logits) return NULL;

    double th0 = get_time_us();
    rmsnorm(s->x, x, w->rms_final_weight, dim);
    double th1 = get_time_us();
    prof->us_lm_head_norm += (th1 - th0);

    quantize(&s->xq, s->x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    double th2 = get_time_us();
    prof->us_lm_head_matmul += (th2 - th1);

    prof->count++;
    return s->logits;
}

typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size; ProbIndex* probindex; float temperature; float topp; unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0; float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

int compare_prob(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (float)(n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler* sampler, float* logits) {
    if (sampler->temperature == 0.0f) return sample_argmax(logits, sampler->vocab_size);
    for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *gpt2_path = "tokenizer_gpt2.bin";
    float temperature = 0.8f, topp = 0.9f;
    int steps = 64;
    int target_prefill_len = 0;
    char *prompt = "Once upon a time";
    unsigned long long rng_seed = (unsigned int)time(NULL);

    if (argc >= 2) { checkpoint_path = argv[1]; }
    else {
        fprintf(stderr, "Usage: %s <checkpoint.q8.bin> [-g <tokenizer_gpt2.bin>] [-i <prompt>] [-b <prefill_len>] [-t <temp>] [-p <topp>] [-n <steps>]\n", argv[0]);
        return 1;
    }

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') break;
        if (argv[i][1] == 't') temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's') rng_seed = atoi(argv[i + 1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'b') target_prefill_len = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'g') gpt2_path = argv[i + 1];
        else if (argv[i][1] == 'z') gpt2_path = argv[i + 1];
    }

    Transformer transformer;
    read_checkpoint(checkpoint_path, &transformer.config, &transformer.weights, &transformer.fd, &transformer.data, &transformer.file_size);

    GPT2Tokenizer gtok = gpt2_load(gpt2_path);

    int max_prompt_cap = target_prefill_len > 0 ? target_prefill_len + 64 : (int)strlen(prompt) + 128;
    int *prompt_tokens = (int*)malloc((size_t)max_prompt_cap * sizeof(int));
    int num_prompt_tokens = 0;
    gpt2_encode(&gtok, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { prompt_tokens[0] = gtok.eot; num_prompt_tokens = 1; }

    // If target_prefill_len is requested (e.g. 32), pad/repeat tokens to reach exactly target_prefill_len
    if (target_prefill_len > 0) {
        int orig_tokens = num_prompt_tokens;
        while (num_prompt_tokens < target_prefill_len) {
            prompt_tokens[num_prompt_tokens] = prompt_tokens[num_prompt_tokens % orig_tokens];
            num_prompt_tokens++;
        }
        num_prompt_tokens = target_prefill_len;
    }

    int total_capacity = num_prompt_tokens + steps + 64;
    transformer.config.seq_len = total_capacity;
    malloc_run_state(&transformer.state, &transformer.config);

    Sampler sampler;
    sampler.vocab_size = transformer.config.vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = rng_seed;
    sampler.probindex = (ProbIndex*)malloc(sampler.vocab_size * sizeof(ProbIndex));

    { int blen0; const unsigned char* b0 = gpt2_token_bytes(&gtok, prompt_tokens[0], &blen0);
      fwrite(b0, 1, blen0, stdout); }

    StepProfile prof_decode = {0};
    StepProfile prof_prefill = {0};

    double ttft_start_us = get_time_us();
    float* logits = NULL;
    for (int pos = 0; pos < num_prompt_tokens; pos++) {
        logits = forward_profiled(&transformer, prompt_tokens[pos], pos, pos == num_prompt_tokens - 1, &prof_prefill);
    }
    double ts0 = get_time_us();
    int next = sample(&sampler, logits);
    double ts1 = get_time_us();
    prof_prefill.us_sample += (ts1 - ts0);
    double ttft_end_us = get_time_us();
    double total_ttft_ms = (ttft_end_us - ttft_start_us) / 1000.0;

    int pos = num_prompt_tokens;
    if (next != gtok.eot) {
        int blen; const unsigned char* b = gpt2_token_bytes(&gtok, next, &blen);
        fwrite(b, 1, blen, stdout); fflush(stdout);
        int token = next;
        while (pos < (steps + num_prompt_tokens)) {
            logits = forward_profiled(&transformer, token, pos, 1, &prof_decode);
            double tsa = get_time_us();
            next = sample(&sampler, logits);
            double tsb = get_time_us();
            prof_decode.us_sample += (tsb - tsa);
            pos++;
            if (next == gtok.eot) break;
            blen = 0; b = gpt2_token_bytes(&gtok, next, &blen);
            fwrite(b, 1, blen, stdout); fflush(stdout);
            token = next;
        }
    }
    printf("\n");

    // Print PREFILL / TTFT REPORT
    double prefill_attn = prof_prefill.us_attn_norm + prof_prefill.us_attn_qkv + prof_prefill.us_attn_rope + prof_prefill.us_attn_core + prof_prefill.us_attn_proj;
    double prefill_mlp = prof_prefill.us_mlp_norm + prof_prefill.us_mlp_gate_val + prof_prefill.us_mlp_act + prof_prefill.us_mlp_down;
    double prefill_lm_head = prof_prefill.us_lm_head_norm + prof_prefill.us_lm_head_matmul;
    double prefill_tok_s = (double)num_prompt_tokens / (total_ttft_ms / 1000.0);

    fprintf(stderr, "\n================ SMOLLM2-135M PREFILL / TTFT REPORT =========================\n");
    fprintf(stderr, "Prefill Prompt Length (Tokens): %d\n", num_prompt_tokens);
    fprintf(stderr, "Total Time to First Token (TTFT): %.3f ms (%.2f prefill tok/s)\n", total_ttft_ms, prefill_tok_s);
    fprintf(stderr, "------------------------------------------------------------------------------\n");
    fprintf(stderr, "Sub-Component               | Time (ms)  | Time (us)    | Portion of TTFT\n");
    fprintf(stderr, "----------------------------+------------+--------------+----------------\n");
    fprintf(stderr, "1. Embedding Lookups (%dx)   | %8.3f   | %10.1f   | %6.2f%%\n", num_prompt_tokens, prof_prefill.us_embed / 1000.0, prof_prefill.us_embed, (prof_prefill.us_embed / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "2. Attention Across Layers   | %8.3f   | %10.1f   | %6.2f%%\n", prefill_attn / 1000.0, prefill_attn, (prefill_attn / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - RMSNorm & Quant        | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_attn_norm / 1000.0, prof_prefill.us_attn_norm, (prof_prefill.us_attn_norm / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - QKV Matmuls (%dx)       | %8.3f   | %10.1f   | %6.2f%%\n", num_prompt_tokens, prof_prefill.us_attn_qkv / 1000.0, prof_prefill.us_attn_qkv, (prof_prefill.us_attn_qkv / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - RoPE Rotation          | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_attn_rope / 1000.0, prof_prefill.us_attn_rope, (prof_prefill.us_attn_rope / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - Softmax & Attn Score   | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_attn_core / 1000.0, prof_prefill.us_attn_core, (prof_prefill.us_attn_core / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - Output Projection      | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_attn_proj / 1000.0, prof_prefill.us_attn_proj, (prof_prefill.us_attn_proj / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "3. FeedForward Network (FFN) | %8.3f   | %10.1f   | %6.2f%%\n", prefill_mlp / 1000.0, prefill_mlp, (prefill_mlp / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - Gate + Value Matmul    | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_mlp_gate_val / 1000.0, prof_prefill.us_mlp_gate_val, (prof_prefill.us_mlp_gate_val / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - GeGLU Activation       | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_mlp_act / 1000.0, prof_prefill.us_mlp_act, (prof_prefill.us_mlp_act / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "   - Down Projection        | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_mlp_down / 1000.0, prof_prefill.us_mlp_down, (prof_prefill.us_mlp_down / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "4. Final LM Head (Pos=%d)    | %8.3f   | %10.1f   | %6.2f%%\n", num_prompt_tokens - 1, prefill_lm_head / 1000.0, prefill_lm_head, (prefill_lm_head / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "5. First Token Sampling     | %8.3f   | %10.1f   | %6.2f%%\n", prof_prefill.us_sample / 1000.0, prof_prefill.us_sample, (prof_prefill.us_sample / (total_ttft_ms * 1000.0)) * 100.0);
    fprintf(stderr, "==============================================================================\n");

    int decode_steps = prof_decode.count;
    if (decode_steps > 0) {
        double avg_embed = prof_decode.us_embed / decode_steps;
        double avg_attn_norm = prof_decode.us_attn_norm / decode_steps;
        double avg_attn_qkv = prof_decode.us_attn_qkv / decode_steps;
        double avg_attn_rope = prof_decode.us_attn_rope / decode_steps;
        double avg_attn_core = prof_decode.us_attn_core / decode_steps;
        double avg_attn_proj = prof_decode.us_attn_proj / decode_steps;
        double avg_attn_total = avg_attn_norm + avg_attn_qkv + avg_attn_rope + avg_attn_core + avg_attn_proj;

        double avg_mlp_norm = prof_decode.us_mlp_norm / decode_steps;
        double avg_mlp_gate_val = prof_decode.us_mlp_gate_val / decode_steps;
        double avg_mlp_act = prof_decode.us_mlp_act / decode_steps;
        double avg_mlp_down = prof_decode.us_mlp_down / decode_steps;
        double avg_mlp_total = avg_mlp_norm + avg_mlp_gate_val + avg_mlp_act + avg_mlp_down;

        double avg_lm_head_norm = prof_decode.us_lm_head_norm / decode_steps;
        double avg_lm_head_matmul = prof_decode.us_lm_head_matmul / decode_steps;
        double avg_lm_head_total = avg_lm_head_norm + avg_lm_head_matmul;

        double avg_sample = prof_decode.us_sample / decode_steps;
        double total_tpot_us = avg_embed + avg_attn_total + avg_mlp_total + avg_lm_head_total + avg_sample;
        double total_tpot_ms = total_tpot_us / 1000.0;
        double tok_per_sec = 1000000.0 / total_tpot_us;

        double pct_embed = (avg_embed / total_tpot_us) * 100.0;
        double pct_attn = (avg_attn_total / total_tpot_us) * 100.0;
        double pct_mlp = (avg_mlp_total / total_tpot_us) * 100.0;
        double pct_lm_head = (avg_lm_head_total / total_tpot_us) * 100.0;
        double pct_sample = (avg_sample / total_tpot_us) * 100.0;

        fprintf(stderr, "\n================ SMOLLM2-135M DECODE REPORT (TPOT) ==========================\n");
        fprintf(stderr, "Total Decode Tokens Measured : %d\n", decode_steps);
        fprintf(stderr, "Average TPOT (Time Per Token): %.3f ms (%.2f tok/s)\n", total_tpot_ms, tok_per_sec);
        fprintf(stderr, "------------------------------------------------------------------------------\n");
        fprintf(stderr, "Sub-Component               | Time (ms)  | Time (us)    | Portion of TPOT\n");
        fprintf(stderr, "----------------------------+------------+--------------+----------------\n");
        fprintf(stderr, "1. Embedding Lookup         | %8.3f   | %10.1f   | %6.2f%%\n", avg_embed / 1000.0, avg_embed, pct_embed);
        fprintf(stderr, "2. Attention Layers (30x)   | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_total / 1000.0, avg_attn_total, pct_attn);
        fprintf(stderr, "   - RMSNorm & Quant        | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_norm / 1000.0, avg_attn_norm, (avg_attn_norm / total_tpot_us) * 100.0);
        fprintf(stderr, "   - QKV Matmuls            | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_qkv / 1000.0, avg_attn_qkv, (avg_attn_qkv / total_tpot_us) * 100.0);
        fprintf(stderr, "   - RoPE Rotation          | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_rope / 1000.0, avg_attn_rope, (avg_attn_rope / total_tpot_us) * 100.0);
        fprintf(stderr, "   - Softmax & Attn Score   | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_core / 1000.0, avg_attn_core, (avg_attn_core / total_tpot_us) * 100.0);
        fprintf(stderr, "   - Output Projection      | %8.3f   | %10.1f   | %6.2f%%\n", avg_attn_proj / 1000.0, avg_attn_proj, (avg_attn_proj / total_tpot_us) * 100.0);
        fprintf(stderr, "3. FeedForward (FFN) (30x)  | %8.3f   | %10.1f   | %6.2f%%\n", avg_mlp_total / 1000.0, avg_mlp_total, pct_mlp);
        fprintf(stderr, "   - Gate + Value Matmul    | %8.3f   | %10.1f   | %6.2f%%\n", avg_mlp_gate_val / 1000.0, avg_mlp_gate_val, (avg_mlp_gate_val / total_tpot_us) * 100.0);
        fprintf(stderr, "   - GeGLU Activation       | %8.3f   | %10.1f   | %6.2f%%\n", avg_mlp_act / 1000.0, avg_mlp_act, (avg_mlp_act / total_tpot_us) * 100.0);
        fprintf(stderr, "   - Down Projection        | %8.3f   | %10.1f   | %6.2f%%\n", avg_mlp_down / 1000.0, avg_mlp_down, (avg_mlp_down / total_tpot_us) * 100.0);
        fprintf(stderr, "4. LM Head (576 -> 50.2k)   | %8.3f   | %10.1f   | %6.2f%% <--- [LM HEAD PORTION]\n", avg_lm_head_total / 1000.0, avg_lm_head_total, pct_lm_head);
        fprintf(stderr, "   - Final RMSNorm          | %8.3f   | %10.1f   | %6.2f%%\n", avg_lm_head_norm / 1000.0, avg_lm_head_norm, (avg_lm_head_norm / total_tpot_us) * 100.0);
        fprintf(stderr, "   - Classifier Matmul      | %8.3f   | %10.1f   | %6.2f%%\n", avg_lm_head_matmul / 1000.0, avg_lm_head_matmul, (avg_lm_head_matmul / total_tpot_us) * 100.0);
        fprintf(stderr, "5. Logits Sampling          | %8.3f   | %10.1f   | %6.2f%%\n", avg_sample / 1000.0, avg_sample, pct_sample);
        fprintf(stderr, "==============================================================================\n");
    }

    free(prompt_tokens); free(sampler.probindex);
    gpt2_free(&gtok); free_transformer(&transformer);
    return 0;
}
