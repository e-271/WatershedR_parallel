 // [[Rcpp::depends(RcppParallel)]]
#include <vector>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <time.h>
#include <Rcpp.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace RcppParallel;

// Create matrix of dimension 2^(number_of_dimensions) X number_of_dimensions
// Each row of this matrix summarizes all possible binary values that CRF can take on
std::vector<std::vector<double>> extract_all_binary_combinations_parallel(int n) {
  // Initialize output matrix
  int nTemp = (int)pow(2, n) - 1;
  std::vector<std::vector<double>> combo_mat(nTemp + 1, std::vector<double>(n));
  
  // Loop through all possible values the CRF can take on 
  for (int i = 0; i <= nTemp; i++) {
    // Loop through dimensions
    for (int k = 0; k < n; k++) {
      if ((i >> k) & 0x1){
        combo_mat[i][k] = 1;
      } else {
        combo_mat[i][k] = 0;
      }
    }
  }
  return combo_mat;
}

// Create matrix of dimension 2^(number_of_dimensions-1) X number_of_dimensions
// Each row of this matrix summarizes all possible binary values that CRF can take on assuming the values of in column column_to_ignore are fixed to be 1
std::vector<std::vector<double>> extract_all_binary_combinations_ignoring_one_row_parallel(int n, int column_to_ignore) {
  // Ignoring column_to_ignore, get matrix of dimension 2^(number_of_dimensions-1) X (number_of_dimensions -1) that representing all possible values CRF can take on for remaining dimensions
  std::vector<std::vector<double>> all_binary_combinations_matrix = extract_all_binary_combinations_parallel(n-1);
  
  // Initialize output matrix
  int N = all_binary_combinations_matrix.size();
  std::vector<std::vector<double>> combo_mat(N, std::vector<double>(n));
  // Loop through all possible values the CRF can take on 
  for (int row_num = 0; row_num < N; row_num++) {
    int counter = 0;
    // Loop through each dimension
    for (int column_num = 0; column_num < n; column_num++) {
      // If the dimension is the column_to_ignore, set it eqaul to one
      if (column_num == column_to_ignore) {
        combo_mat[row_num][column_num] = 1;
        // If the column is not a column to ignore (ie a dimension we are marginalizing out)
      } else {
        combo_mat[row_num][column_num] = all_binary_combinations_matrix[row_num][counter];
        counter += 1;
      }
    }
  }
  return combo_mat;
}

// Create matrix of dimension 2^(number_of_dimensions-2) X number_of_dimensions
// Each row of this matrix summarizes all possible binary values that CRF can take on assuming the values of column_to_ignore1 and column_to_ignore2 are fixed to be 1
std::vector<std::vector<double>> extract_all_binary_combinations_ignoring_two_row_parallel(int n, int column_to_ignore1, int column_to_ignore2) {
  // Special case (handeled seperately) for when there are only two dimensions
  if (n == 2) {
    std::vector<std::vector<double>> combo_mat(1,std::vector<double>(2));
    combo_mat[0][0] = 1;
    combo_mat[0][1] = 1;
    return combo_mat;
    // General case for when the number of dimensions is greater than 2
  } else {
    // Ignoring column_to_ignore1 and column_to_ignore2, get matrix of dimension 2^(number_of_dimensions-2) X (number_of_dimensions -2) that representing all possible values CRF can take on for remaining dimensions
    std::vector<std::vector<double>> all_binary_combinations_matrix = extract_all_binary_combinations_parallel(n-2);
    // Initialize output matrix
    int nrow = all_binary_combinations_matrix.size();
    std::vector<std::vector<double>> combo_mat(nrow, std::vector<double>(n));
    // Loop through all possible values CRF can take on
    for (int row_num = 0; row_num < nrow; row_num++) {
      int counter = 0;
      // Loop through each dimension
      for (int column_num = 0; column_num < n; column_num++) {
        // If the dimension is a column to ignore, set it equal to one
        if (column_num == column_to_ignore1 || column_num == column_to_ignore2) {
          combo_mat[row_num][column_num] = 1;
          // If the column is not a column to ignore (ie a dimension we are marginalizing out)
        } else {
          combo_mat[row_num][column_num] = all_binary_combinations_matrix[row_num][counter];
          counter += 1;
        }
      }
    }
    return combo_mat;
  }
}

// For a CRF value (particular row in all_binary_combinations_matrix), compute the relative CRF weight
double un_normalized_crf_weight_parallel(std::vector<std::vector<double>>& all_binary_combinations_matrix, const int combination_number, const RMatrix<double>& feat, const RMatrix<double>& discrete_outliers, const RVector<double>& theta_singleton, const RMatrix<double>& theta_pair, const RMatrix<double>& theta, const RMatrix<double>& phi_inlier, const RMatrix<double>& phi_outlier, const int number_of_dimensions, int sample_num, bool posterior_bool) {
  // Initialize weight
  double weight = 0;
  int dimension_counter = 0;
  // Loop through dimensions
  for (int dimension=0; dimension < number_of_dimensions; dimension++) {
    // Add term from intercept
    weight += all_binary_combinations_matrix[combination_number][dimension]*theta_singleton[dimension];
    // Loop through features
    for (int d = 0; d < feat.ncol(); d++) {
      // Add term from features
      weight += all_binary_combinations_matrix[combination_number][dimension]*feat(sample_num,d)*theta(d,dimension);
    }
    // Loop through all pairs of dimensions
    for (int dimension2=dimension; dimension2 < number_of_dimensions; dimension2++) {
      if (dimension != dimension2) {
        // Add edge weight
        weight += all_binary_combinations_matrix[combination_number][dimension]*all_binary_combinations_matrix[combination_number][dimension2]*theta_pair(0, dimension_counter);
        dimension_counter += 1;
      }
    }
    // Check to see if we are supposed to incorperate expression data && whether the expression data is observed
    if (posterior_bool == true && discrete_outliers(sample_num, dimension) == discrete_outliers(sample_num, dimension)) {
      if (all_binary_combinations_matrix[combination_number][dimension] == 1) {
        weight += log(phi_outlier(dimension, discrete_outliers(sample_num, dimension) - 1));
      } else {
        weight += log(phi_inlier(dimension, discrete_outliers(sample_num, dimension) - 1));
      }
    }
  }
  return weight;
}

// Compute CRF normalization constant for a specifc sample
double exact_normalization_constant_parallel(const RMatrix<double>& feat, const RMatrix<double>& discrete_outliers, const RVector<double>& theta_singleton, const RMatrix<double>& theta_pair, const RMatrix<double>& theta, const RMatrix<double>& phi_inlier, const RMatrix<double>& phi_outlier, int number_of_dimensions, int sample_num, bool posterior_bool) {
  // Extract matrix summarizing all possible values the CRF can take on
  // Create matrix of dimension 2^(number_of_dimensions) X number_of_dimensions
  // Each row of this matrix summarizes all possible binary values that CRF can take on
  std::vector<std::vector<double>> all_binary_combinations_matrix = extract_all_binary_combinations_parallel(number_of_dimensions);
  // Initialize variable keep track of normalization constant
  double val = 0;
  // Loop through each possible value the CRF can take on 
  for (int combination_number = 0; combination_number < all_binary_combinations_matrix.size(); combination_number++) {
    // And compute un-normalized CRF weight corresponding to that sample
    double un_normalized_wight = un_normalized_crf_weight_parallel(all_binary_combinations_matrix, combination_number, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, posterior_bool);
    val += exp(un_normalized_wight);
  }
  return log(val);
}

// Compute probability of a CRF label (Z*) 
// ie compute P(Z=Z*)
double exact_probability_parallel(double normalization_constant, const RMatrix<double>& feat, const RMatrix<double>& discrete_outliers, const RVector<double>& theta_singleton, const RMatrix<double>& theta_pair, const RMatrix<double>& theta, const RMatrix<double>& phi_inlier, const RMatrix<double>& phi_outlier, const int number_of_dimensions, int sample_num, int combination_number, std::vector<std::vector<double>>& all_binary_combinations_ignoring_one_row, const bool posterior_bool) {
  double prob = exp(un_normalized_crf_weight_parallel(all_binary_combinations_ignoring_one_row, combination_number, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, posterior_bool) - normalization_constant);
  return prob;
}

// Compute marginal posterior probability for this sample in this dimension
// Compute P(Z_dimension=1|G) if posterior_bool==false
// Compute P(Z_dimension=1|G,E) if posterior_bool==true
// Involves marginalizing out all other dimensions
double exact_marginal_probability_parallel(double normalization_constant, const RMatrix<double>& feat, const RMatrix<double>& discrete_outliers, const RVector<double>& theta_singleton, const RMatrix<double>& theta_pair, const RMatrix<double>& theta, const RMatrix<double>& phi_inlier, const RMatrix<double>& phi_outlier, const int number_of_dimensions, int sample_num, int dimension, const bool posterior_bool) {
  // Create matrix of dimension 2^(number_of_dimensions-1) X number_of_dimensions
  // Each row of this matrix summarizes all possible binary values that CRF can take on assuming the values of in column column_to_ignore are fixed to be 1
  std::vector<std::vector<double>> all_binary_combinations_ignoring_one_row = extract_all_binary_combinations_ignoring_one_row_parallel(number_of_dimensions, dimension);
  double marginal_prob = 0;
  // Loop through each of the rows (possible CRF values) of the above matrix
  for (int combination_number = 0; combination_number < all_binary_combinations_ignoring_one_row.size(); combination_number++) {
    // For this CRF value, compute the probability
    marginal_prob += exact_probability_parallel(normalization_constant, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, combination_number, all_binary_combinations_ignoring_one_row, posterior_bool);
  }
  return marginal_prob;
}


// Compute marginal pairwise probability for dimension pair for a particular sample
// Compute P(Z_dimension1=1, Z_dimension2=1| G) if posterior_bool==false
// Compute P(Z_dimension1=1, Z_dimension2=1| G,E) if posterior_bool==true
// Involves marginalizing out all other dimensions
double exact_marginal_pairwise_probability_parallel(double normalization_constant, int dimension1, int dimension2, int dimension_counter, const RMatrix<double>& feat, const RMatrix<double>& discrete_outliers, const RVector<double>& theta_singleton, const RMatrix<double>& theta_pair, const RMatrix<double>& theta, const RMatrix<double>& phi_inlier, const RMatrix<double>& phi_outlier, const int number_of_dimensions, int sample_num, bool posterior_bool) {
  // Initialize marginal probability variable
  double marginal_prob = 0;
  // Create matrix of dimension 2^(number_of_dimensions-2) X number_of_dimensions
  // Each row of this matrix summarizes all possible binary values that CRF can take on assuming the values of dimension_1 and dimension_2 are fixed to be 1
  std::vector<std::vector<double>> marginal_binary_combinations_matrix = extract_all_binary_combinations_ignoring_two_row_parallel(number_of_dimensions, dimension1, dimension2);
  // Loop through each of the rows (possible CRF values) of the above matrix
  for (int combination_number = 0; combination_number < marginal_binary_combinations_matrix.size(); combination_number++) {
    // For this CRF value, compute the probability
    marginal_prob += exact_probability_parallel(normalization_constant, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, combination_number, marginal_binary_combinations_matrix, posterior_bool);
  }
  return marginal_prob;
}

// RcppParallel worker struct
struct MarginalPosteriors : public Worker
{   
   // Inputs of length N
   const RMatrix<double> feat;
   const RMatrix<double> discrete_outliers;
   // Model weights
   const RVector<double> theta_singleton;
   const RMatrix<double> theta_pair;
   const RMatrix<double> theta;
   const RMatrix<double> phi_inlier; 
   const RMatrix<double> phi_outlier;
   // Parameters
   const int number_of_dimensions;
   const int number_of_pairs; 
   const bool posterior_bool;
   // destination matrix
   RMatrix<double> probabilities;
   RMatrix<double> probabilities_pairwise;

   // constructors
   MarginalPosteriors(const NumericMatrix feat,
                const NumericMatrix discrete_outliers, 
                const NumericVector theta_singleton, 
                const NumericMatrix theta_pair, 
                const NumericMatrix theta, 
                const NumericMatrix phi_inlier, 
                const NumericMatrix phi_outlier, 
                const int number_of_dimensions, 
                const int number_of_pairs, 
                const bool posterior_bool,
                NumericMatrix probabilities,
                NumericMatrix probabilities_pairwise) : 
                              feat(feat), discrete_outliers(discrete_outliers), 
                              theta_singleton(theta_singleton), theta_pair(theta_pair), theta(theta), 
                              phi_inlier(phi_inlier), phi_outlier(phi_outlier), number_of_dimensions(number_of_dimensions), 
                              number_of_pairs(number_of_pairs), posterior_bool(posterior_bool), 
                              probabilities(probabilities), probabilities_pairwise(probabilities_pairwise) {}
   
   // accumulate just the element of the range I've been asked to
   void operator()(std::size_t begin, std::size_t end) {
    // START OP
    // Loop through samples
    for (std::size_t sample_num = begin; sample_num < end; sample_num++) {
      // Compute normalization constant for this sample
      double normalization_constant = exact_normalization_constant_parallel(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, posterior_bool);
      // Initialize variable to help when iterating through pairs of edges
      int dimension_counter = 0;
      // Loop through dimensions
      for (int dimension = 0; dimension < number_of_dimensions; dimension++) {
        // Compute marginal posterior probability for this sample in this dimension
        probabilities(sample_num, dimension) = exact_marginal_probability_parallel(normalization_constant, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, dimension, posterior_bool);
        // Loop through pairs of dimensions
        for (int dimension2=dimension; dimension2 < number_of_dimensions; dimension2++) {
          if (dimension != dimension2) {
            // Compute marginal pairwise probability for dimension pair
            probabilities_pairwise(sample_num, dimension_counter) = exact_marginal_pairwise_probability_parallel(normalization_constant, dimension, dimension2, dimension_counter, feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, sample_num, posterior_bool);
            dimension_counter += 1;
          }
        }
      }
    }
    //END OP
   }
};


// Compute both marginal Posterior probability as well as marginal pairwise probabilities for Conditional random field using exact inference
// If posterior_bool==true, compute P(Z|E,G)
// If posterior_bool==false, compute P(Z|G)
// [[Rcpp::export]]
List update_marginal_probabilities_exact_inference_parallel_cpp(NumericMatrix feat, NumericMatrix discrete_outliers, NumericVector theta_singleton, NumericMatrix theta_pair, NumericMatrix theta, NumericMatrix phi_inlier, NumericMatrix phi_outlier, int number_of_dimensions, int number_of_pairs, bool posterior_bool) {
  // Initialize output matrices
  NumericMatrix probabilities(feat.nrow(), number_of_dimensions);
  NumericMatrix probabilities_pairwise(feat.nrow(), number_of_pairs);

  // Create MarginalPosteriors functor 
  MarginalPosteriors marginalPosteriors(feat, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, number_of_pairs, posterior_bool, probabilities, probabilities_pairwise);

  // call parallelFor to do the work
  parallelFor(0, feat.nrow(), marginalPosteriors);

  // TODO cast returned RMatrix<double> to a NumericMatrix here
  List ret;
  ret["probability"] = probabilities;
  ret["probability_pairwise"] = probabilities_pairwise;
  return ret;
}


// RcppParallel worker struct
struct CRFLikelihood : public Worker
{   
   // Inputs of length N
   const RMatrix<double> feat;
   const RMatrix<double> posterior;
   const RMatrix<double> posterior_pairwise;
   const RMatrix<double> discrete_outliers;
   // Model weights
   const RVector<double> theta_singleton;
   const RMatrix<double> theta_pair;
   const RMatrix<double> theta;
   const RMatrix<double> phi_inlier; 
   const RMatrix<double> phi_outlier;
   // Parameters
   const int number_of_dimensions;
   const double lambda;
   const double lambda_pair;
   const double lambda_singleton;
   
   // destination vector
   RVector<double> log_likelihood;
   
   // constructors
   CRFLikelihood(const NumericMatrix feat,
                const NumericMatrix posterior, 
                const NumericMatrix posterior_pairwise, 
                const NumericMatrix discrete_outliers, 
                const NumericVector theta_singleton, 
                const NumericMatrix theta_pair, 
                const NumericMatrix theta, 
                const NumericMatrix phi_inlier, 
                const NumericMatrix phi_outlier, 
                const int number_of_dimensions, 
                const double lambda, 
                const double lambda_pair, 
                const double lambda_singleton,
                NumericVector log_likelihood) : 
                              feat(feat), posterior(posterior), posterior_pairwise(posterior_pairwise), discrete_outliers(discrete_outliers), 
                              theta_singleton(theta_singleton), theta_pair(theta_pair), theta(theta), 
                              phi_inlier(phi_inlier), phi_outlier(phi_outlier), number_of_dimensions(number_of_dimensions), 
                              lambda(lambda), lambda_pair(lambda_pair), lambda_singleton(lambda_singleton), log_likelihood(log_likelihood) {}
    CRFLikelihood(const CRFLikelihood& crfLikelihood, Split) : 
        feat(crfLikelihood.feat),
        posterior(crfLikelihood.posterior),
        posterior_pairwise(crfLikelihood.posterior_pairwise),
        discrete_outliers(crfLikelihood.discrete_outliers),
        theta_singleton(crfLikelihood.theta_singleton),
        theta_pair(crfLikelihood.theta_pair),
        theta(crfLikelihood.theta),
        phi_inlier(crfLikelihood.phi_inlier),
        phi_outlier(crfLikelihood.phi_outlier),
        number_of_dimensions(crfLikelihood.number_of_dimensions),
        lambda(crfLikelihood.lambda),
        lambda_pair(crfLikelihood.lambda_pair),
        lambda_singleton(crfLikelihood.lambda_singleton),
        log_likelihood(crfLikelihood.log_likelihood) {}
   
   // accumulate just the element of the range I've been asked to
   void operator()(std::size_t begin, std::size_t end) {
    // log_likelihood += std::accumulate(feat.begin() + begin, feat.begin() + end, 0.0);

    // START OP
    for (int sample_num = begin; sample_num < end; sample_num++) {
      // Compute normalization constant for this sample
      double normalization_constant = exact_normalization_constant_parallel(feat, 
                                                                            discrete_outliers, 
                                                                            theta_singleton, 
                                                                            theta_pair, 
                                                                            theta, 
                                                                            phi_inlier, 
                                                                            phi_outlier, 
                                                                            number_of_dimensions, 
                                                                            sample_num, 
                                                                            false);
      // Subtract normalization constant from the log likelihood
      log_likelihood[sample_num] = -normalization_constant;
      // Loop through dimensions
      //int dimension = 0;
      int dimension_counter = 0;
      for (int dimension = 0; dimension < number_of_dimensions; dimension++) {
        // Add contribution of intercept term for this dimension
        log_likelihood[sample_num] += theta_singleton[dimension]*posterior(sample_num, dimension);
        // Add contribution of feature terms for this dimension
        for (int d = 0; d < feat.ncol(); d++) {
          log_likelihood[sample_num] += theta(d, dimension)*feat(sample_num, d)*posterior(sample_num, dimension);
        }
        // This nested for loop will loop through all pairs of dimensions
        for (int dimension2=dimension; dimension2 < number_of_dimensions; dimension2++) {
          if (dimension != dimension2) {
            // Add contribution of a given edge pair
            log_likelihood[sample_num] += theta_pair(0, dimension_counter)*posterior_pairwise(sample_num, dimension_counter);
            dimension_counter += 1;
          }
        }
      }
    }
    //END OP
   }
     
   // join my value with that of another Sum
  //  void join(const CRFLikelihood& rhs) { 
  //     log_likelihood += rhs.log_likelihood; 
  //  }
};



// Compute exact likelihood of K=number_of_dimensions dimensionsal Conditional Random Field (CRF)
// [[Rcpp::export]]
double compute_crf_likelihood_exact_inference_parallel_cpp(NumericMatrix posterior, NumericMatrix posterior_pairwise, NumericMatrix feat, NumericMatrix discrete_outliers, NumericVector theta_singleton, NumericMatrix theta_pair, NumericMatrix theta, NumericMatrix phi_inlier, NumericMatrix phi_outlier, int number_of_dimensions, double lambda, double lambda_pair, double lambda_singleton) {
  // Initialize output likelihood
  // double log_likelihood = 0;
  NumericVector sample_log_likelihood(feat.nrow());

  // CRF functor 
  CRFLikelihood crfLikelihood(feat, posterior, posterior_pairwise, discrete_outliers, theta_singleton, theta_pair, theta, phi_inlier, phi_outlier, number_of_dimensions, lambda, lambda_pair, lambda_singleton, sample_log_likelihood);

  // call parallelReduce to do the work
  // parallelReduce(0, feat.nrow(), crfLikelihood);
  // log_likelihood = crfLikelihood.log_likelihood;
  // call parallelFor to do the work
  parallelFor(0, feat.nrow(), crfLikelihood);

  // sum over log likelihoods (this is done after parallelism to avoid non-deterministic results from parallelReduce)
  double log_likelihood = 0;
  for (int i=0; i < sample_log_likelihood.size(); i++) {
    log_likelihood += sample_log_likelihood[i];
  }

  // Normalize log likelihood by the number of samples
  log_likelihood = log_likelihood/feat.nrow();
  
  // Add L2 penalties
  int dimension_counter = 0;
  for (int dimension = 0; dimension < number_of_dimensions; dimension++) {
    // Generally do not regularize intercept terms (ie lambda_singleton is usually set to zero)
    log_likelihood = log_likelihood - .5*lambda_singleton*(theta_singleton(dimension)*theta_singleton(dimension));
    // Regularize edge weights
    for (int dimension2=dimension; dimension2 < number_of_dimensions; dimension2++) {
      if (dimension != dimension2) {
        log_likelihood = log_likelihood - .5*lambda_pair*(theta_pair(0,dimension_counter)*theta_pair(0, dimension_counter));
        dimension_counter += 1;
      }
    }
    // Regularize feature vectors 
    for (int d = 0; d < feat.ncol(); d++) {
      log_likelihood = log_likelihood - .5*lambda*(theta(d,dimension)*theta(d,dimension));
    }
  }
  return log_likelihood;
}