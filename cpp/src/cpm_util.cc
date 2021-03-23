#include <iostream>
#include <vector>
#include <armadillo>
#include <algorithm>


using namespace arma;


/*
 * Creates dim x dim tridiagonal matrix representing
 * the Laplacian Beltrami operator
*/

sp_mat createLaplacian(const std::vector<int>& band, const int N, const int grid_width, const double dx) {
    const double diag = -4 / (dx * dx);
    const double off = 1 / (dx * dx);

    sp_mat lap (band.size(), N);

    for (int i = 0; i < band.size(); i++) {
        const int idx = band[i];
        lap(i, idx) = diag;
        lap(i, idx - 1) = lap(i, idx + 1) = off;
        lap(i, idx - grid_width) = lap(i, idx + grid_width) = off;
    }

    // Do the trimming
    sp_mat trim_lap (band.size(), band.size());

    for (int i = 0; i < band.size(); i++) {
        const int idx = band[i];
        trim_lap.col(i) = lap.col(idx);
    }

    return trim_lap;
}


/*
 * Find k closest items to val in arr
 * Assumes that arr is uniform; ie arr[i] = arr[i] + i * (arr[1] - arr[0]) 
 * Returns index of first element in the list of k, so the k closest are
 * arr[left], arr[left + 1], ... , arr[left + k - 1]
*/
int kClosest(const std::vector<double>& arr, const double val, const int k = 4) {
    // Find value to the left and right of val
    int left = (val - arr[0]) / (arr[1] - arr[0]);
    int right = left + 1;

    for (int count = 0; count < k; count++) {
        if (left < 0) {
            right++;
        } else if (right >= arr.size()) {
            left--;
        } else if (val - arr[left] <= arr[right] - val) {
            left--;
        } else {
            right++;
        }
    }
    return left + 1;
}


// Lagrange weight
double lagrange1D(const double x, const std::vector<double>& arr, const int i) {
    double result = 1;
    for (int j = 0; j < arr.size(); j++) {
        if (i == j) {
            continue;
        }
        result *= (x - arr[j]) / (arr[i] - arr[j]);
    }
    return result;
}




// Create interpolation matrix
sp_mat createInterpMatrix(const std::vector<double>& x_pts, const std::vector<double>& y_pts, const std::vector<vec>& pts, const std::vector<int>& band) {
    sp_mat E(pts.size(), band.size());

    for (int k = 0; k < pts.size(); k++) {
        const vec p = pts[k];

        // Find points enclosed in interpolation stencil
        constexpr int K = 4;
        const int x_start = kClosest(x_pts, p(0), K);
        const int y_start = kClosest(y_pts, p(1), K);

        std::vector<double> x_stencil (K);
        std::vector<double> y_stencil (K);

        for (int i = 0; i < K; i++) {
            x_stencil[i] = x_pts[x_start + i];
            y_stencil[i] = y_pts[y_start + i];
        }

        // Compute and place weight
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                const double w = lagrange1D(p(0), x_stencil, i) * lagrange1D(p(1), y_stencil, j);
                const int pts_idx = x_pts.size() * (y_start + j) + (x_start + i);
                const auto it = std::lower_bound(band.begin(), band.end(), pts_idx);
                const int band_k = it - band.begin();
                E(k, band_k) = w;
            }
        }
        const auto f = sum(E.row(k));
        if ( f > 1.01 || f < 0.99) {
            throw "sum of row in interpolation matrix is not 1";
        }
    }
    return E;
}


// Uses Jacobi iteration to find solution
vec jacobiSolve(const sp_mat& E, const sp_mat& L, const vec& b, vec u) {
    // Manually take the inverse of diag(L) as there is no inv(sp_mat)
    sp_mat diagInv (L.n_rows, L.n_cols);
    for (int i = 0; i < L.n_rows; i++) {
        diagInv(i, i) = 1 / L(i, i);
    }

    const sp_mat M = E * diagInv;
    const sp_mat woDiag = L - diagmat(L);

    // Begin Jacobi Iteration
    constexpr double goal = 0.00000000001;
    double delta = 1;
    int k = 0;
    constexpr int maxSteps = 10000;

    while (k < maxSteps && delta > goal) {
        const vec unew = M * (b - woDiag * u);
        delta = norm(unew - u);
        u = unew;
        k++;
    }

    return u;
}



