/* Minimal float32 causal convolution + exact float64 log-CTC for inspection.
 * No trained parameters are embedded. CTC uses host exp/log1p without a beam.
 */
extern double exp(double);
extern double log1p(double);
static double plus(double a, double b) {
    if (a == -__builtin_inf()) return b;
    if (b == -__builtin_inf()) return a;
    double high = a > b ? a : b, low = a > b ? b : a;
    return high + log1p(exp(low - high));
}
__attribute__((visibility("default")))
void conv(const float *x, const float *w, const float *b, float *y,
          int ci, int co, int k, int dilation, int frames, int relu) {
    int length = (k - 1) * dilation + frames;
    for (int o = 0; o < co; ++o) {
        for (int t = 0; t < frames; ++t) {
            float sum = b[o];
            for (int i = 0; i < ci; ++i)
                for (int j = 0; j < k; ++j)
                    sum += x[i * length + t + j * dilation]
                         * w[(o * ci + i) * k + j];
            y[o * frames + t] = relu && sum < 0.0f ? 0.0f : sum;
        }
    }
}
__attribute__((visibility("default")))
void ctc(const int *parent, const int *label, const double *lp,
         double *pb, double *pn, double *qb, double *qn, double *totals,
         int nodes, int symbols) {
    for (int i = 0; i < nodes; ++i) totals[i] = plus(pb[i], pn[i]);
    for (int i = 0; i < nodes; ++i) {
        qb[i] = totals[i] + lp[0];
        if (i == 0) { qn[i] = -__builtin_inf(); continue; }
        int p = parent[i];
        double previous = label[i] == label[p] ? pb[p] : totals[p];
        qn[i] = plus(pn[i], previous) + lp[label[i]];
    }
}
