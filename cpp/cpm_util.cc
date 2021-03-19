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


// Utility printer for debugging
// TODO remove this when converting to library?
template <typename T>
void print(const T x) {
    std::cout << x << std::endl;
}


// Create interpolation matrix
// TODO add assert for row sum
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
        auto unew = M * (b - woDiag * u);
        delta = norm(unew - u);
        u = unew;
        k++;
    }

    return u;
}

// TODO init and f functions are temporary below - remove later
double init(const double x, const double y) {
    const double angle = atan2(y, x);
    return sin(angle);
}

double f(const double x, const double y) {
    const double angle = atan2(y, x);
    return -sin(angle) - 144 * sin(12 * angle);
}


int main() {
    wall_clock timer;
    timer.tic();

    // Create grid
    constexpr double dx = 0.1;
    std::vector<double> x_pts;
    std::vector<double> y_pts;
    double t = -2;
    while (t < 2 + dx) {
        x_pts.push_back(t);
        y_pts.push_back(t);
        t += dx;
    }

    // Compute closest point
    std::vector<vec> cp_pts;
    std::vector<int> band;

    int count = 0;
    for (int i = 0; i < x_pts.size(); i++) {
        for (int j = 0; j < y_pts.size(); j++) {
            vec p = {x_pts[j], y_pts[i] };
            vec cp_p = p;

            if (x_pts[i] != 0 || y_pts[j] != 0) {
                cp_p /= norm(p);
            } else {
                cp_p = {1, 0};
            }

            double d = norm(cp_p - p);
            if (d <= 0.36059) {
                band.push_back(count);
                cp_pts.push_back(cp_p);
            }
            count++;
        }
    }

    sp_mat laplacian = createLaplacian(band, x_pts.size() * y_pts.size(), x_pts.size(), dx);
    sp_mat E = createInterpMatrix(x_pts, y_pts, cp_pts, band);

    // Compute right hand side and initial guess u_0 based on
    // u(theta) = sin(theta) + sin(12 theta)
    // f = - sin(theta) - 144 sin(12 theta)
    vec b (band.size());
    vec u_0 (band.size());
    for (int idx = 0; idx < band.size(); idx++) {
        int j = band[idx] / x_pts.size();
        int i = band[idx] % x_pts.size();

        b(idx) = f(x_pts[i], y_pts[j]);
        u_0(idx) = init(x_pts[i], y_pts[j]);
    }


    // Solve
    auto result = jacobiSolve(E, laplacian, b, u_0);
    auto s = timer.toc();
    std::cout << "Time: " << s << std::endl;
}

