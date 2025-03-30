#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of cols
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(ind){
      ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
    } else {
      ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
    }
    if(!ind){
      throw std::runtime_error("Could not allocate ind array.");
    }
    if(val){
      val = (val_t*) realloc(val, sizeof(val_t) * nnz);
    } else {
      val = (val_t*) malloc(sizeof(val_t) * nnz);
    }
    if(!val){
      throw std::runtime_error("Could not allocate val array.");
    }
    this->nrows = nrows;
  }

  csr_t ( const csr_t &other)
  {
    this->nrows = this->ncols = 0;
    this->ptr = nullptr;
    this->ind = nullptr;
    this->val = nullptr;
    this->reserve(other.nrows, other.ptr[other.nrows]);
    memcpy(ptr, other.ptr, sizeof(ptr_t) * (nrows+1));
    memcpy(ind, other.ind, sizeof(idx_t) * ptr[nrows]);
    memcpy(val, other.val, sizeof(val_t) * ptr[nrows]);
    this->ncols = other.ncols;
  }

  /** 
   * Transpose matrix
   */
  void transpose()
  {
    auto nptr = (ptr_t*) calloc(ncols+1, sizeof(ptr_t));
    auto nind = (idx_t*) malloc(sizeof(idx_t) * ptr[nrows]);
    auto nval = (val_t*) malloc(sizeof(val_t) * ptr[nrows]);
    if(!nptr || !nind || !nval){
      throw std::runtime_error("Could not allocate memory for transpose.");
    }
    // count number of non-zeros in each column
    for(ptr_t i=0; i < ptr[nrows]; ++i){
      nptr[ind[i]+1]++;
    }
    // compute prefix sum
    for(idx_t i=1; i <= ncols; ++i){
      nptr[i] += nptr[i-1];
    }
    // fill in the values
    for(idx_t i=0; i < nrows; ++i){
      for(ptr_t j=ptr[i]; j < ptr[i+1]; ++j){
        auto col = ind[j];
        auto idx = nptr[col]++;
        nind[idx] = i;
        nval[idx] = val[j];
      }
    }
    // fix the pointer array
    for(idx_t i=ncols; i > 0; --i){
      nptr[i] = nptr[i-1];
    }
    nptr[0] = 0;
    free(ptr);
    free(ind);
    free(val);
    ptr = nptr;
    ind = nind;
    val = nval;
    std::swap(nrows, ncols);
  }

  /**
   * Create random matrix with given sparsity factor.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    // create a vector of random numbers of size nrows
    vector<double> numbers(nrows);
    std::random_device r;
    std::default_random_engine gen(r());
    std::uniform_int_distribution<int> uniform_dist(1, ncols * factor);
    std::generate(numbers.begin(), numbers.end(), [&] {return uniform_dist(gen); });
    auto sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);
    // normalize the vector to sum to 1
    for (idx_t i = 0; i < nrows; i++) {
      numbers[i] = numbers[i] / (double) sum;
    }
    /* fill in ptr array; generate random row sizes */
    for(idx_t i=0; i < mat->nrows; ++i){
      mat->ptr[i+1] = mat->ptr[i] + std::max((ptr_t)(numbers[i] * nnz), (ptr_t)1);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/RAND_MAX);
        }
      }
    }

    return mat;
  }

  /* check if matrix is equal to the other matrix */
  bool equal(csr_t * other, const double eps = 1e-3)
  {
    if(nrows != other->nrows || ncols != other->ncols || ptr[nrows] != other->ptr[other->nrows]){
      return false;
    }
    // matrices need to be sorted to check equality
    if(!this->indices_sorted()){
      this->sort_indices();
    }
    if(!other->indices_sorted()){
      other->sort_indices();
    }
    for(idx_t i=0; i < nrows; ++i){
      for(ptr_t j=ptr[i]; j < ptr[i+1]; ++j){
        if(ind[j] != other->ind[j]){
          cout << "row " << i << " cid missmatch: " << ind[j] << " != " << other->ind[j] << endl;
          return false;
        }
        if(abs(val[j] - other->val[j]) > eps){
          cout << "row " << i << " val missmatch: " << val[j] << " != " << other->val[j] << endl;
          return false;
        }
      }
    }
    return true;
  }

  /* sort non-zeros in each row in increasing order of columnd ids */
  void sort_indices()
  {
    std::vector<std::pair<idx_t, val_t>> row;
    #pragma omp parallel for firstprivate(row)
    for(idx_t i = 0; i < nrows; ++i) {
      for(ptr_t j = ptr[i]; j < ptr[i + 1]; ++j) {
          row.push_back(std::make_pair(ind[j], val[j]));
      }
      sort(row.begin(), row.end());
      for(ptr_t j = ptr[i]; j < ptr[i + 1]; ++j) {
        ind[j] = row[j-ptr[i]].first;
        val[j] = row[j-ptr[i]].second;
      }
      row.clear();
    }
  }

  /* check if indices are sorted */
  bool indices_sorted()
  {
    for(idx_t i=0; i < nrows; ++i){
      for(ptr_t j=ptr[i]; j < ptr[i+1]-1; ++j){
        if(ind[j+1] <= ind[j]){
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Write matrix to text file
   * @param output_fpath File to write to
   */
  void write(const std::string output_fpath, const bool header=false)
  {
    std::fstream resfile;
    resfile.open(output_fpath, std::ios::out);
    if(!resfile){
      throw std::runtime_error("Could not open output file for writing.");
    }
    if(header){
      resfile << nrows << " " << ncols << " " << ptr[nrows] << std::endl;
    }
    for(idx_t i=0; i < nrows; ++i){
      for(ptr_t j=ptr[i]; j < ptr[i+1]; ++j){
        resfile << ind[j] << " " << val[j];
        if(j+1 < ptr[i+1]){
          resfile << " ";
        }
      }
      resfile << std::endl;
    }
    resfile.close();
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat) {
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

/**
 * Multiply A and B and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B  Matrix B.
 * @param C  Output matrix
 */
void sparsematmult_sparse_sparse(csr_t * A, csr_t * B, csr_t *C, idx_t tile_size=10)
{
  C->ncols = B->ncols;
  // first, we need to transpose B in order to access its vectors by column
  B->transpose();
  // for the sparse-sparse dot product method to work, we need to sort the non-zeors in each row of A 
  // by column id and in each column of B by row id
  A->sort_indices();
  B->sort_indices();
  // create initial C space
  ptr_t maxnnz = B->nrows * 5; // max number of non-zeros in C
  ptr_t nnz = 0; // current number of non-zeors in C

  C->ncols = B->nrows;
  C->reserve(A->nrows, maxnnz);
  #pragma omp parallel
  {
    // space for thread to keep temporary results
    vector<pair<idx_t, val_t>> results;
    results.reserve(tile_size * B->nrows); // it should never be more than this
    idx_t * ncands = (idx_t *) calloc(tile_size+1, sizeof(idx_t));
    if(!ncands){
      throw std::runtime_error("Could not allocate cands pointer array for thread.");
    }
    #pragma omp for ordered schedule(static, 1)
    for(idx_t i=0; i < A->nrows; i+=tile_size){
      for(idx_t t=0; t<tile_size && i+t < A->nrows; ++t){
        idx_t it = i+t;
        for(idx_t c=0; c < B->nrows; ++c){
          double v = 0;
          for(ptr_t j=A->ptr[it], k=B->ptr[c]; j < A->ptr[it+1] && k < B->ptr[c+1]; ){
            if(A->ind[j] == B->ind[k]){ // same column ID
              v += (double) A->val[j] * B->val[k];
              j++;
              k++;
            } else if(A->ind[j] < B->ind[k]){
              j++;
            } else {
              k++;
            }
          }
          if(v != 0){
            results.push_back(make_pair(c, v));
          }
        }
        ncands[t+1] = results.size();
      }
      #pragma omp ordered
      {
        for(idx_t t=0; t<tile_size && i+t < A->nrows; ++t){
          if(nnz + ncands[t+1] - ncands[t] > maxnnz){
            maxnnz *= 2;
            C->reserve(A->nrows, maxnnz);
          }
          for(idx_t r=ncands[t]; r < ncands[t+1]; ++r){
            C->ind[nnz] = results[r].first;
            C->val[nnz++] = results[r].second;
          }
          C->ptr[i+t+1] = C->ptr[i+t] + ncands[t+1] - ncands[t];
        }
        results.clear();
      }
    }
    free(ncands);
  }
  if(nnz > 0){
    C->reserve(A->nrows, nnz);
  }
}

/**
 * Multiply A and B and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B  Matrix B.
 * @param C  Output matrix
 */
void sparsematmult_accum(csr_t * A, csr_t * B, csr_t *C)
{
  val_t val_accum[B->ncols];
  idx_t accum_index[B->ncols];
  ptr_t nnz = 0;
  ptr_t local_nnz = 0;
  C->ncols = B->ncols;
  C->reserve(A->nrows, 0);

  #pragma omp parallel for
  for(idx_t i = 0; i < B->ncols; ++i) { // Set up and zero the accumulator
    val_accum[i] = 0;
    accum_index[i] = 0;
  }
  
  // Iterate through all rows of A
  #pragma omp parallel for ordered schedule(static, 1) shared(nnz) firstprivate(val_accum, accum_index, local_nnz)
  for(idx_t i = 0; i < A->nrows; ++i) { // Iterate thru rows
    for(idx_t j = A->ptr[i]; j < A->ptr[i+1]; ++j) { // Iterate thru values within row
      idx_t a_col = A->ind[j];
      for(idx_t k = B->ptr[a_col]; k < B->ptr[a_col + 1]; ++k) { // Iterate thru values in row in matrix B
        idx_t b_col = B->ind[k]; 
        val_t prod = A->val[j] * B->val[k];
        
        // Check if it is alr accum'd, if not, new entry in both index and accumulator
        if(val_accum[b_col] != 0.0) {
          val_accum[b_col] = prod;
        }
        else {
          val_accum[b_col] = prod; 
          accum_index[local_nnz] = b_col;
          ++local_nnz;
        }
      }
    }

    #pragma omp ordered
    {
      // Update the new number of non-zero values
      nnz += local_nnz;

      // Use updated value to resize the C matrix
      if(local_nnz > 0 ) C->reserve(A->nrows, nnz);

      // Copy accumulated values and indicies into C
      for(idx_t index = 0; index < local_nnz; ++index) {
        C->ind[nnz + index - local_nnz] = accum_index[index];
        C->val[nnz + index - local_nnz] = val_accum[accum_index[index]];
        
        // Clear accum back to 0
        val_accum[accum_index[index]] = 0;
        accum_index[index] = 0;
      }
      // Update pointer array
      C->ptr[i + 1] = nnz;
    }
    // Zero the local_nnz value
    local_nnz = 0;
  }
  C->ptr[A->nrows] = nnz;
  
}

int main(int argc, char *argv[])
{
  if(argc < 4){
    cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);
  int nthreads = 1;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }
  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_nrows: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols, ncols2, factor); // Note B is not transposed.
  test_matrix(A);
  test_matrix(B);
  cout << A->info("A") << endl;
  cout << B->info("B") << endl;

  auto A1 = new csr_t(*A);
  auto B1 = new csr_t(*B);
  auto C1 = new csr_t(); // Note that C has no data allocations so far.
  auto t1 = omp_get_wtime();
  sparsematmult_sparse_sparse(A1, B1, C1);
  auto t2 = omp_get_wtime();
  cout << C1->info("C1") << endl;
  cout << "Execution time (sparse-sparse dot-product): " << t2-t1 << endl;
  delete A1;
  delete B1;

  auto A2 = new csr_t(*A);
  auto B2 = new csr_t(*B);
  auto C2 = new csr_t(); // Note that C has no data allocations so far.
  t1 = omp_get_wtime();
  sparsematmult_accum(A2, B2, C2);
  t2 = omp_get_wtime();
  cout << C2->info("C2") << endl;
  cout << "Execution time (accumulator): " << t2-t1 << endl;
  delete A2;
  delete B2;

  cout << "C1 == C2: " << (C1->equal(C2) ? "true" : "false") << endl;

  delete A;
  delete B;
  delete C1;
  delete C2;

  return 0;
}
