#include <armadillo>
#include <vector>
#include <iostream>
#include "cpm_util.h"

using namespace arma;

// TODO init and f functions are temporary below - remove later
double init(const double x, const double y) {
    const double angle = atan2(y, x);
    return sin(angle);
}

double f(const double x, const double y) {
    const double angle = atan2(y, x);
    return -sin(angle) - 144 * sin(12 * angle);
}


// Utility printer for debugging
// TODO remove this when converting to library?
template <typename T>
void print(const T x) {
    std::cout << x << std::endl;
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
