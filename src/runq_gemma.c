/* Inference for Gemma-3 / Gemma models in pure C, INT8 Q8_0 quantized forward pass.
 *
 * Tailored specifically for Gemma architecture:
 * 1. Gemma Tokenizer: BOS = 2 (<bos>), EOS = 1 (<eos>), PAD = 0 (<pad>), SentencePiece prefix U+2581 (" ").
 * 2. Gemma Embedding Scaling: Multiplies input token embeddings by sqrt(dim).
 * 3. GeGLU Activation: 0.5 * x * (1 + erf(x / sqrt(2)))
 * 4. Memory Efficiency: On-the-fly token dequantization (no 671MB table malloc) + dynamic KV-cache sizing.
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

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension (640 for Gemma-3 270M)
    int hidden_dim; // for ffn layers (2048)
    int n_layers; // number of layers (18)
    int n_heads; // number of query heads (4)
    int n_kv_heads; // number of key/value heads (1)
    int vocab_size; // vocabulary size (262144)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s;     // scaling factors
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens; // (vocab_size, dim) quantized token embeddings
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    float* rms_final_weight; // (dim,)
    QuantizedTensor *wcls; // classifier
} TransformerWeights;

typedef struct {
    float *x; // activation at current time stamp (dim,)
    float *xb; // inside residual branch (dim,)
    float *xb2; // convenience buffer (dim,)
    float *hb; // hidden dimension in ffn (hidden_dim,)
    float *hb2; // hidden dimension in ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (kv_dim,)
    float *v; // value (kv_dim,)
    float *att; // scores/attention values (n_heads, seq_len)
    float *logits; // output logits (vocab_size,)
    float *key_cache;   // (n_layers, seq_len, kv_dim)
    float *value_cache; // (n_layers, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

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
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize_token_embedding(QuantizedTensor *q_tokens, float* x, int token, int dim) {
    size_t offset = (size_t)token * dim;
    float embed_scale = sqrtf((float)dim); // Gemma multiplies embedding by sqrt(dim)
    for (int i = 0; i < dim; i++) {
        size_t idx = offset + i;
        x[i] = (q_tokens->q[idx] * q_tokens->s[idx / GS]) * embed_scale;
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
    if (!file) { fprintf(stderr, "Couldn't open checkpoint file %s\n", checkpoint); exit(EXIT_FAILURE); }
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
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if (t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Neural net forward pass

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        // Gemma RMSNorm uses x * (1.0 + weight) or x * weight depending on normalization baseline
        o[j] = x[j] * ss * weight[j];
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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

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

float* forward(Transformer* transformer, int token, int pos, int compute_logits) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    dequantize_token_embedding(w->q_tokens, x, token, dim);

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);

        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE rotation
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

        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }

        rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // GeGLU activation
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val = 0.5f * val * (1.0f + erff(val * 0.70710678118654752440f));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }

    if (!compute_logits) return NULL;

    rmsnorm(s->x, x, w->rms_final_weight, dim);
    quantize(&s->xq, s->x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Gemma SentencePiece Tokenizer Implementation

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc((size_t)vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc((size_t)vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) break;
        if (fread(&len, sizeof(int), 1, file) != 1) break;
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) break;
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i]) free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size || !t->vocab[token]) return "";
    char *piece = t->vocab[token];
    return piece;
}

void safe_printf_gemma(char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    // Replace SentencePiece U+2581 (0xE2 0x96 0x81) with a regular space ' '
    const unsigned char *p = (const unsigned char*)piece;
    while (*p) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {
            putchar(' ');
            p += 3;
        } else if (p[0] == '<' && strncmp((const char*)p, "<0x", 3) == 0 && p[5] == '>') {
            unsigned int byte_val = 0;
            if (sscanf((const char*)p, "<0x%02X>", &byte_val) == 1) {
                if (isprint(byte_val) || isspace(byte_val)) putchar(byte_val);
            }
            p += 6;
        } else {
            putchar(*p);
            p++;
        }
    }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode_gemma(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc((size_t)t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i] ? t->vocab[i] : "";
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    *n_tokens = 0;
    // Gemma: BOS is Token 2 (<bos>)
    if (bos) tokens[(*n_tokens)++] = 2;

    if (text[0] == '\0') return;

    // SentencePiece encodes space as "\xe2\x96\x81"
    char* str_buffer = (char*)malloc((size_t)(t->max_token_length * 2 + 16));
    
    // Add leading space prefix if text doesn't start with space
    int dummy_prefix = str_lookup("\xe2\x96\x81", t->sorted_vocab, t->vocab_size);
    
    // Character by character BPE encoding
    for (char *c = text; *c != '\0'; c++) {
        char single_char[8] = {0};
        if (*c == ' ') {
            strcpy(single_char, "\xe2\x96\x81");
        } else {
            single_char[0] = *c;
            single_char[1] = '\0';
        }
        int id = str_lookup(single_char, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // Byte fallback
            char byte_tok[16];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)*c);
            id = str_lookup(byte_tok, t->sorted_vocab, t->vocab_size);
            if (id != -1) tokens[(*n_tokens)++] = id;
        }
    }

    // Merge pass (BPE pair merges by highest score)
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, t->max_token_length * 2 + 16, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    free(str_buffer);
    if (eos) tokens[(*n_tokens)++] = 1; // Gemma: EOS is Token 1 (<eos>)
}

// ----------------------------------------------------------------------------
// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
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
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    if (sampler->temperature == 0.0f) {
        return sample_argmax(logits, sampler->vocab_size);
    }
    for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

// ----------------------------------------------------------------------------
// Generation Loop

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 0.8f;
    float topp = 0.9f;
    int steps = 128;
    char *prompt = "Once upon a time";
    unsigned long long rng_seed = (unsigned int)time(NULL);

    if (argc >= 2) { checkpoint_path = argv[1]; }
    else {
        fprintf(stderr, "Usage: %s <checkpoint.q8.bin> [-z <tokenizer.bin>] [-i <prompt>] [-t <temp>] [-p <topp>] [-n <steps>]\n", argv[0]);
        return 1;
    }

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') break;
        if (argv[i][1] == 't') temperature = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 's') rng_seed = atoi(argv[i + 1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
    }

    Transformer transformer;
    read_checkpoint(checkpoint_path, &transformer.config, &transformer.weights, &transformer.fd, &transformer.data, &transformer.file_size);
    
    // Dynamic KV-cache allocation sized to requested steps
    if (steps > 0) {
        transformer.config.seq_len = steps;
    } else {
        steps = transformer.config.seq_len;
    }
    malloc_run_state(&transformer.state, &transformer.config);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    sampler.vocab_size = transformer.config.vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = rng_seed;
    sampler.probindex = (ProbIndex*)malloc(sampler.vocab_size * sizeof(ProbIndex));

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((size_t)(strlen(prompt) + 16) * sizeof(int));
    encode_gemma(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    long t_start = time_in_ms();
    float* logits = NULL;
    for (int pos = 0; pos < num_prompt_tokens; pos++) {
        logits = forward(&transformer, prompt_tokens[pos], pos, pos == num_prompt_tokens - 1);
    }
    long t_prefill_end = time_in_ms();
    int next = sample(&sampler, logits);

    int pos = num_prompt_tokens;
    int token = next;

    char* piece = decode(&tokenizer, prompt_tokens[0], next);
    safe_printf_gemma(piece);
    fflush(stdout);

    while (pos < steps) {
        if (next == 1) break; // Gemma: EOS token = 1
        logits = forward(&transformer, token, pos, 1);
        next = sample(&sampler, logits);
        pos++;
        piece = decode(&tokenizer, token, next);
        safe_printf_gemma(piece);
        fflush(stdout);
        token = next;
    }
    printf("\n");

    long t_end = time_in_ms();
    double prefill_ms = (double)(t_prefill_end - t_start);
    int decode_count = pos > num_prompt_tokens ? (pos - num_prompt_tokens) : 0;
    double decode_duration_ms = (double)(t_end - t_prefill_end);
    double decode_tok_s = decode_count > 0 && decode_duration_ms > 0 ? (decode_count / decode_duration_ms * 1000.0) : 0.0;
    double total_tok_s = (pos > 1 && (t_end - t_start) > 0) ? ((pos - 1) / (double)(t_end - t_start) * 1000.0) : 0.0;

    fprintf(stderr, "prefill_tokens: %d, prefill_ms: %.2f, ttft_ms: %.2f, decode_tokens: %d, decode_tok_s: %.2f, achieved tok/s: %.6f\n",
            num_prompt_tokens, prefill_ms, prefill_ms, decode_count, decode_tok_s, total_tok_s);

    free(prompt_tokens);
    free(sampler.probindex);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
