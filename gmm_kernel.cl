__constant sampler_t gmmSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Wytłumacz argumenty kernela
__kernel void gmm_kernel(
    __read_only image2d_t inputImage,
    __global float *means,
    __global float *vars,
    __global float *weights,
    __global uchar *mask,
    const int K,
    const float alpha,
    const float bg_thresh,
    const float match_thresh,
    const float init_var,
    const float min_var,
    const int first_frame)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int W = get_global_size(0);
    const int H = get_global_size(1);
    if (x >= W || y >= H)
        return;

    const int pixel_number = y * W + x;
    const int base = pixel_number * K;

    uint4 px = read_imageui(inputImage, gmmSampler, (int2)(x, y));
    const float val = 0.114f * convert_float(px.x)
                    + 0.587f * convert_float(px.y)
                    + 0.299f * convert_float(px.z);

    if (first_frame)
    {
        for (int k = 0; k < K; k++)
        {
            means[base + k] = val;
            vars[base + k] = init_var;
            weights[base + k] = (k == 0) ? 1.0f : 0.0f;
        }
        // uchar to jeden bajt
        mask[pixel_number] = (uchar)0;
        return;
    }

    int hit = -1;
    for (int k = 0; k < K; k++)
    {
        float sd = sqrt(vars[base + k]);
        if (fabs(val - means[base + k]) <= match_thresh * sd)
        {
            hit = k;
            break;
        }
    }

    if (hit >= 0)
    {
        float mu = means[base + hit];
        float va = vars[base + hit];
        float diff = val - mu;
        float rho = alpha;
        // Aktualizacja średnich i wariancji
        means[base + hit] = mu + rho * diff;
        float new_var = va + rho * (diff * diff - va);
        vars[base + hit] = (new_var < min_var) ? min_var : new_var;
        // Aktualizacja wag
        for (int k = 0; k < K; k++)
            weights[base + k] *= (1.0f - alpha);
        weights[base + hit] += alpha;
    }
    else
    {
        // Aktualizacja wag
        for (int k = 0; k < K; k++) {
            weights[base + k] *= (1.0f - alpha);
        }
        // Znalezienie najsłabszego komponentu
        int weak = 0;
        float worst = weights[base] / sqrt(vars[base]);
        for (int k = 1; k < K; k++)
        {
            float rank = weights[base + k] / sqrt(vars[base + k]);
            if (rank < worst)
            {
                worst = rank;
                weak = k;
            }
        }
        // Wymiana najsłabszego komponentu
        means[base + weak] = val;
        vars[base + weak] = init_var;
        weights[base + weak] = alpha;
    }

    // Normalizacja wag
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += weights[base + k];
    if (sum > 1e-9f)
    {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++)
            weights[base + k] *= inv;
    }

    // Sortowanie rozkładów gaussa, od "najważniejszego"
    for (int i = 0; i < K - 1; i++)
    {
        for (int j = 0; j < K - i - 1; j++)
        {
            float rj = weights[base + j] / sqrt(vars[base + j]);
            float rj1 = weights[base + j + 1] / sqrt(vars[base + j + 1]);
            if (rj < rj1)
            {
                float t;
                t = means[base + j];
                means[base + j] = means[base + j + 1];
                means[base + j + 1] = t;
                t = vars[base + j];
                vars[base + j] = vars[base + j + 1];
                vars[base + j + 1] = t;
                t = weights[base + j];
                weights[base + j] = weights[base + j + 1];
                weights[base + j + 1] = t;
            }
        }
    }

    // Sprawdzenie czy piksel należy do tła
    int is_bg = 0;
    if (hit >= 0)
    {
        float cumulative = 0.0f;
        int B = 1;
        for (int k = 0; k < K; k++)
        {
            cumulative += weights[base + k];
            B = k + 1;
            if (cumulative > bg_thresh)
            {
                break;
            }
        }
        for (int k = 0; k < B; k++)
        {
            float sd = sqrt(vars[base + k]);
            if (fabs(val - means[base + k]) <= match_thresh * sd)
            {
                is_bg = 1;
                break;
            }
        }
    }
    mask[pixel_number] = is_bg ? (uchar)0 : (uchar)255;
}
