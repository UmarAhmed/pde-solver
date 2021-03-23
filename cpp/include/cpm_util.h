#ifndef CPM_UTIL_H
#define CPM_UTIL_H

#include <armadillo>
#include <vector>

/*
 * Implementation of closest point method functions
*/

arma::sp_mat createLaplacian(const std::vector<int>& band, const int N, const int grid_width, const double dx);



arma::sp_mat createInterpMatrix(const std::vector<double>& x_pts, const std::vector<double>& y_pts, const std::vector<arma::vec>& pts, const std::vector<int>& band);



arma::vec jacobiSolve(const arma::sp_mat& E, const arma::sp_mat& L, const arma::vec& b, arma::vec u);

#endif
