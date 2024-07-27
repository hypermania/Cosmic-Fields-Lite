#include "equations_cuda.cuh"

// #include <thrust/host_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/reduce.h>
// #include <thrust/functional.h>
// #include <thrust/fill.h>
// #include <thrust/transform.h>

#include "cuda_wrapper.cuh"
#include "fdm3d_cuda.cuh"

typedef thrust::device_vector<double> state_type;

__global__
void KG_FRW_equation_kernel(const double *x, double *dxdt,
			    const double m,
			    const double H_t, const double inv_ah_sqr,
			    const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  
  dxdt[IDX_OF(N, a, b, c)] = x[N*N*N+IDX_OF(N, a, b, c)];
  dxdt[N*N*N+IDX_OF(N, a, b, c)] =
    (-3.0) * H_t * x[N*N*N+IDX_OF(N, a, b, c)]
    - m * m * x[IDX_OF(N, a, b, c)]
    + inv_ah_sqr * (-6.0 * x[IDX_OF(N, a, b, c)]
		    + x[IDX_OF(N, (a+1)%N, b, c)]
		    + x[IDX_OF(N, (a+N-1)%N, b, c)]
		    + x[IDX_OF(N, a, (b+1)%N, c)]
		    + x[IDX_OF(N, a, (b+N-1)%N, c)]
		    + x[IDX_OF(N, a, b, (c+1)%N)]
		    + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}

__global__
void Lambda_FRW_equation_kernel(const double *x, double *dxdt,
				const double m, const double lambda,
				const double H_t, const double inv_ah_sqr,
				const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  dxdt[idx1] = x[idx2];
  dxdt[idx2] =
    (-3.0) * H_t * x[idx2]
    - m * m * x[idx1]
    - lambda * x[idx1] * x[idx1] * x[idx1]
    + inv_ah_sqr * (-6.0 * x[idx1]
		    + x[IDX_OF(N, (a+1)%N, b, c)]
		    + x[IDX_OF(N, (a+N-1)%N, b, c)]
		    + x[IDX_OF(N, a, (b+1)%N, c)]
		    + x[IDX_OF(N, a, (b+N-1)%N, c)]
		    + x[IDX_OF(N, a, b, (c+1)%N)]
		    + x[IDX_OF(N, a, b, (c+N-1)%N)]);
  
  // dxdt[IDX_OF(N, a, b, c)] = x[N*N*N+IDX_OF(N, a, b, c)];
  // dxdt[N*N*N+IDX_OF(N, a, b, c)] =
  //   (-3.0) * H_t * x[N*N*N+IDX_OF(N, a, b, c)]
  //   - m * m * x[IDX_OF(N, a, b, c)]
  //   - lambda * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)]
  //   + inv_ah_sqr * (-6.0 * x[IDX_OF(N, a, b, c)]
  // 		    + x[IDX_OF(N, (a+1)%N, b, c)]
  // 		    + x[IDX_OF(N, (a+N-1)%N, b, c)]
  // 		    + x[IDX_OF(N, a, (b+1)%N, c)]
  // 		    + x[IDX_OF(N, a, (b+N-1)%N, c)]
  // 		    + x[IDX_OF(N, a, b, (c+1)%N)]
  // 		    + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}


/*
  Evolve equation with V(\varphi) = m^2 f_a^2 (sqrt(1 + \varphi^2 / f_a^2) - 1).
  \ddot{\varphi} + 3 H \dot{\varphi} - \frac{\nabla^2}{a^2} \varphi 
  + m^2 \varphi / sqrt(1 + \varphi^2 / f_a^2) = 0
*/
__global__
void Sqrt_Potential_FRW_equation_kernel(const double *x, double *dxdt,
					const double m, const double f_a,
					const double H_t, const double inv_ah_sqr,
					const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  dxdt[idx1] = x[idx2];
  dxdt[idx2] =
    (-3.0) * H_t * x[idx2]
    - m * m * x[idx1] / sqrt(1.0 + x[idx1] * x[idx1] / (f_a * f_a) )
    + inv_ah_sqr * (-6.0 * x[idx1]
		    + x[IDX_OF(N, (a+1)%N, b, c)]
		    + x[IDX_OF(N, (a+N-1)%N, b, c)]
		    + x[IDX_OF(N, a, (b+1)%N, c)]
		    + x[IDX_OF(N, a, (b+N-1)%N, c)]
		    + x[IDX_OF(N, a, b, (c+1)%N)]
		    + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}


/*
  Evolve equation with fixed curvature perturbation Psi.
  \ddot{\varphi} + 3 H \dot{\varphi} - e^{4\Psi} \frac{\nabla^2}{a^2} \varphi 
  + e^{2\Psi} m^2 \varphi = 0
*/
__global__
void Fixed_Curvature_FRW_equation_kernel(const double *x, double *dxdt,
					 const double *Psi, const double m,
					 const double H_t, const double inv_ah_sqr,
					 const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  dxdt[idx1] = x[idx2];
  dxdt[idx2] =
    (-3.0) * H_t * x[idx2]
    - exp(2 * Psi[idx1]) * m * m * x[idx1]
    + exp(4 * Psi[idx1]) * inv_ah_sqr * (-6.0 * x[idx1]
					 + x[IDX_OF(N, (a+1)%N, b, c)]
					 + x[IDX_OF(N, (a+N-1)%N, b, c)]
					 + x[IDX_OF(N, a, (b+1)%N, c)]
					 + x[IDX_OF(N, a, (b+N-1)%N, c)]
					 + x[IDX_OF(N, a, b, (c+1)%N)]
					 + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}


__global__
void Comoving_Curvature_Psi_fft_kernel(const double *R_fft, double *Psi_fft,
				       const long long int N, const double factor)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  if(c <= half_N) {
    int a_shifted = (a<half_N) ? a : ((int)N-a);
    int b_shifted = (b<half_N) ? b : ((int)N-b);
  
    int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c);

    int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c * c;

    if(s_sqr == 0) {
      Psi_fft[offset_k] = 0;
      Psi_fft[offset_k + 1] = 0;
    } else {
      double omega_eta = sqrt(__int2double_rn(s_sqr)) * factor;

      double sin_val;
      double cos_val;
      sincos(omega_eta, &sin_val, &cos_val);
      double transfer_function = 2.0 * (sin_val - omega_eta * cos_val) / (omega_eta * omega_eta * omega_eta) / (N * N * N);
      Psi_fft[offset_k] = R_fft[offset_k] * transfer_function;
      Psi_fft[offset_k + 1] = R_fft[offset_k + 1] * transfer_function;
    }
  }
}


__global__
void Comoving_Curvature_dPsidt_fft_kernel(const double *R_fft, double *dPsidt_fft,
					  const long long int N, const double factor, const double H_t)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  if(c <= N/2) {
    int a_shifted = (a<half_N) ? a : (N-a);
    int b_shifted = (b<half_N) ? b : (N-b);
  
    int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c);

    int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c * c;

    if(s_sqr == 0) {
      dPsidt_fft[offset_k] = 0;
      dPsidt_fft[offset_k + 1] = 0;
    } else {
      double omega_eta = sqrt(__int2double_rn(s_sqr)) * factor;

      double sin_val;
      double cos_val;
      sincos(omega_eta, &sin_val, &cos_val);
      double transfer_function = H_t * 2.0 * (3 * omega_eta * cos_val + (omega_eta * omega_eta - 3.0) * sin_val) / (omega_eta * omega_eta * omega_eta) / (N * N * N);
      dPsidt_fft[offset_k] = R_fft[offset_k] * transfer_function;
      dPsidt_fft[offset_k + 1] = R_fft[offset_k + 1] * transfer_function;
    }
  }
}


__global__
void Comoving_Curvature_Psi_dPsidt_fft_kernel(const double *R_fft, double *Psi_fft, double *dPsidt_fft,
					      const long long int N, const double factor, const double H_t)
{
  using namespace std::numbers;
  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;
  int c = blockDim.z * blockIdx.z + threadIdx.z;

  int half_N = N/2;
  if(c <= N/2) {
    int a_shifted = (a<half_N) ? a : (N-a);
    int b_shifted = (b<half_N) ? b : (N-b);
  
    int offset_k = 2 * (N*(half_N+1)*a + (half_N+1)*b + c);

    int s_sqr = a_shifted * a_shifted + b_shifted * b_shifted + c * c;

    if(s_sqr == 0) {
      Psi_fft[offset_k] = 0;
      Psi_fft[offset_k + 1] = 0;
      dPsidt_fft[offset_k] = 0;
      dPsidt_fft[offset_k + 1] = 0;
    } else {
      double omega_eta = sqrt(__int2double_rn(s_sqr)) * factor;

      double sin_val;
      double cos_val;
      sincos(omega_eta, &sin_val, &cos_val);
      double common_factor = 2.0 / (omega_eta * omega_eta * omega_eta) / (N * N * N);

      double R_val_r = R_fft[offset_k];
      double R_val_i = R_fft[offset_k + 1];

      double transfer_function_Psi = (sin_val - omega_eta * cos_val) * common_factor;
      Psi_fft[offset_k] = R_val_r * transfer_function_Psi;
      Psi_fft[offset_k + 1] = R_val_i * transfer_function_Psi;

      double transfer_function_dPsidt = H_t * (3 * omega_eta * cos_val + (omega_eta * omega_eta - 3.0) * sin_val) * common_factor;
      dPsidt_fft[offset_k] = R_val_r * transfer_function_dPsidt;
      dPsidt_fft[offset_k + 1] = R_val_i * transfer_function_dPsidt;
    }
  }
}


/*
  Evolve equation with variable curvature perturbation Psi and \dot{\Psi}.
  \ddot{\varphi} + 3 H \dot{\varphi} - (1+4\Psi) \frac{\nabla^2}{a^2} \varphi 
  + (1+2\Psi) m^2 \varphi - 4 \dot{\Psi} \dot{\varphi} = 0
  
  Psi and dPsidt are assumed to be in FFTW padded format.
*/
__global__
void Variable_Curvature_FRW_equation_kernel(const double *x, double *dxdt,
					    const double *Psi, const double *dPsidt,
					    const double m, const double H_t,
					    const double inv_ah_sqr, const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;
  int idx3 = PADDED_IDX_OF(N, a, b, c);

  dxdt[idx1] = x[idx2];
  dxdt[idx2] =
    (-3.0 * H_t + 4.0 * dPsidt[idx3]) * x[idx2]
    - exp(2 * Psi[idx3]) * m * m * x[idx1]
    + exp(4 * Psi[idx3]) * inv_ah_sqr * (-6.0 * x[idx1]
					 + x[IDX_OF(N, (a+1)%N, b, c)]
					 + x[IDX_OF(N, (a+N-1)%N, b, c)]
					 + x[IDX_OF(N, a, (b+1)%N, c)]
					 + x[IDX_OF(N, a, (b+N-1)%N, c)]
					 + x[IDX_OF(N, a, b, (c+1)%N)]
					 + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}


/*
  Evolve equation with variable curvature perturbation Psi and \dot{\Psi}.
  \ddot{\varphi} + 3 H \dot{\varphi} - (1+4\Psi) \frac{\nabla^2}{a^2} \varphi 
  + (1+2\Psi) m^2 \varphi - 4 \dot{\Psi} \dot{\varphi} = 0
  
  Psi and dPsidt are assumed to be in FFTW padded format.

  Approximations are introduced for Psi and the exponential.
*/
__global__
void Approximate_Variable_Curvature_FRW_equation_kernel(const double *x, double *dxdt,
							const double *Psi, const double *dPsidt,
							const double m, const double H_t,
							const double inv_ah_sqr,
							const long long int N, const long long int M)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  int scale_diff = N / M;
  int half_scale_diff = scale_diff / 2;
  int idx3 = PADDED_IDX_OF(M, ((a+half_scale_diff)%N)/scale_diff, ((b+half_scale_diff)%N)/scale_diff, ((c+half_scale_diff)%N)/scale_diff);

  double Psi_val = Psi[idx3];

  dxdt[idx1] = x[idx2];
  dxdt[idx2] =
    (-3.0 * H_t + 4.0 * dPsidt[idx3]) * x[idx2]
    - (1 + 2.0*Psi_val + 2.0 * Psi_val * Psi_val) * m * m * x[idx1]
    + (1 + 4.0*Psi_val + 8.0 * Psi_val * Psi_val) * inv_ah_sqr * (-6.0 * x[idx1]
								  + x[IDX_OF(N, (a+1)%N, b, c)]
								  + x[IDX_OF(N, (a+N-1)%N, b, c)]
								  + x[IDX_OF(N, a, (b+1)%N, c)]
								  + x[IDX_OF(N, a, (b+N-1)%N, c)]
								  + x[IDX_OF(N, a, b, (c+1)%N)]
								  + x[IDX_OF(N, a, b, (c+N-1)%N)]);
}


__global__
void energy_density_kernel(const double *x, double *rho,
			   const double m, const double lambda,
			   const double inv_ah_sqr,
			   const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  double f_val = x[idx1];
  double delta_over_x = x[IDX_OF(N, (a+1)%N, b, c)] - x[IDX_OF(N, (a+N-1)%N, b, c)];
  double delta_over_y = x[IDX_OF(N, a, (b+1)%N, c)] - x[IDX_OF(N, a, (b+N-1)%N, c)];
  double delta_over_z = x[IDX_OF(N, a, b, (c+1)%N)] - x[IDX_OF(N, a, b, (c+N-1)%N)];
  
  rho[idx1] =
    0.5 * x[idx2] * x[idx2]
    + (0.5 / 4.0) * inv_ah_sqr *
    ( delta_over_x * delta_over_x + delta_over_y * delta_over_y + delta_over_z * delta_over_z )
    + 0.5 * m * m * f_val * f_val
    + 0.25 * lambda * f_val * f_val * f_val * f_val;

  // int a = blockIdx.x;
  // int b = blockIdx.y;
  // int c = threadIdx.x;
  
  // rho[IDX_OF(N, a, b, c)] =
  //   0.5 * x[N*N*N+IDX_OF(N, a, b, c)] * x[N*N*N+IDX_OF(N, a, b, c)]
  //   + (0.5 / 4.0) * inv_ah_sqr *
  //   ( pow(x[IDX_OF(N, (a+1)%N, b, c)] - x[IDX_OF(N, (a+N-1)%N, b, c)], 2)
  //     + pow(x[IDX_OF(N, a, (b+1)%N, c)] - x[IDX_OF(N, a, (b+N-1)%N, c)], 2)
  //     + pow(x[IDX_OF(N, a, b, (c+1)%N)] - x[IDX_OF(N, a, b, (c+N-1)%N)], 2) )
  //   + 0.5 * m * m * x[IDX_OF(N, a, b, c)] * x[IDX_OF(N, a, b, c)]
  //   + 0.25 * lambda * pow(x[IDX_OF(N, a, b, c)], 4);

}


__global__
void dot_energy_density_kernel(const double *x, double *dot_rho,
			       const double m,
			       const double inv_ah_sqr, const double H_t,
			       const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  double f_val = x[idx1];
  double delta_over_x = x[IDX_OF(N, (a+1)%N, b, c)] - x[IDX_OF(N, (a+N-1)%N, b, c)];
  double delta_over_y = x[IDX_OF(N, a, (b+1)%N, c)] - x[IDX_OF(N, a, (b+N-1)%N, c)];
  double delta_over_z = x[IDX_OF(N, a, b, (c+1)%N)] - x[IDX_OF(N, a, b, (c+N-1)%N)];
  double dot_delta_over_x = x[N*N*N+IDX_OF(N, (a+1)%N, b, c)] - x[N*N*N+IDX_OF(N, (a+N-1)%N, b, c)];
  double dot_delta_over_y = x[N*N*N+IDX_OF(N, a, (b+1)%N, c)] - x[N*N*N+IDX_OF(N, a, (b+N-1)%N, c)];
  double dot_delta_over_z = x[N*N*N+IDX_OF(N, a, b, (c+1)%N)] - x[N*N*N+IDX_OF(N, a, b, (c+N-1)%N)];
  
  double laplacian = inv_ah_sqr * (-6.0 * x[IDX_OF(N, a, b, c)]
				   + x[IDX_OF(N, (a+1)%N, b, c)]
				   + x[IDX_OF(N, (a+N-1)%N, b, c)]
				   + x[IDX_OF(N, a, (b+1)%N, c)]
				   + x[IDX_OF(N, a, (b+N-1)%N, c)]
				   + x[IDX_OF(N, a, b, (c+1)%N)]
				   + x[IDX_OF(N, a, b, (c+N-1)%N)]);
  double rho =
    0.5 * x[idx2] * x[idx2]
    + (0.5 / 4.0) * inv_ah_sqr *
    ( delta_over_x * delta_over_x + delta_over_y * delta_over_y + delta_over_z * delta_over_z )
    + 0.5 * m * m * f_val * f_val;

  double p =
    0.5 * x[idx2] * x[idx2]
    - 0.5 * m * m * f_val * f_val
    - (1.0 / 4.0 / 6.0) * inv_ah_sqr *
    ( delta_over_x * delta_over_x + delta_over_y * delta_over_y + delta_over_z * delta_over_z );

  double varphi_term = x[idx2] * laplacian + (1.0 / 4.0) * inv_ah_sqr *
    ( dot_delta_over_x * delta_over_x + dot_delta_over_y * delta_over_y + dot_delta_over_z * delta_over_z );
  
  dot_rho[idx1] = varphi_term - 3.0 * H_t * (rho + p);
}

/*
  Computes energy density and stores at *rho.
  \rho = \dot{\varphi}^2 / 2 + (\nabla\varphi)^2 / (2 a^2) 
  + m^2 f_a^2 (\sqrt{1 + \varphi^2 / f_a^2} - 1)
*/
__global__
void Sqrt_Potential_energy_density_kernel(const double *x, double *rho,
					  const double m, const double f_a,
					  const double inv_ah_sqr,
					  const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  double f_val = x[idx1];
  double delta_over_x = x[IDX_OF(N, (a+1)%N, b, c)] - x[IDX_OF(N, (a+N-1)%N, b, c)];
  double delta_over_y = x[IDX_OF(N, a, (b+1)%N, c)] - x[IDX_OF(N, a, (b+N-1)%N, c)];
  double delta_over_z = x[IDX_OF(N, a, b, (c+1)%N)] - x[IDX_OF(N, a, b, (c+N-1)%N)];
  
  rho[idx1] =
    0.5 * x[idx2] * x[idx2]
    + (0.5 / 4.0) * inv_ah_sqr *
    ( delta_over_x * delta_over_x + delta_over_y * delta_over_y + delta_over_z * delta_over_z )
    + m * m * f_a * f_a * (sqrt(1.0 + f_val * f_val / (f_a * f_a)) - 1.0);
}


/*
  Computes energy density and stores at *rho.
  \rho = e^{-2\Psi} \dot{\varphi}^2 / 2 + e^{2\Psi} (\nabla\varphi)^2 / (2 a^2) + m^2 \varphi^2
*/
__global__
void Curvature_energy_density_kernel(const double *x, double *rho,
				     const double *Psi,
				     const double m, const double inv_ah_sqr,
				     const long long int N)
{
  int a = blockIdx.x;
  int b = blockIdx.y;
  int c = threadIdx.x;
  int idx1 = IDX_OF(N, a, b, c);
  int idx2 = N*N*N + idx1;

  double f_val = x[idx1];
  double delta_over_x = x[IDX_OF(N, (a+1)%N, b, c)] - x[IDX_OF(N, (a+N-1)%N, b, c)];
  double delta_over_y = x[IDX_OF(N, a, (b+1)%N, c)] - x[IDX_OF(N, a, (b+N-1)%N, c)];
  double delta_over_z = x[IDX_OF(N, a, b, (c+1)%N)] - x[IDX_OF(N, a, b, (c+N-1)%N)];
  
  rho[idx1] =
    exp(-2 * Psi[idx1]) * 0.5 * x[idx2] * x[idx2]
    + exp(2 * Psi[idx1]) * (0.5 / 4.0) * inv_ah_sqr *
    ( delta_over_x * delta_over_x + delta_over_y * delta_over_y + delta_over_z * delta_over_z )
    + 0.5 * m * m * f_val * f_val;
}


void compute_deriv_test(const Eigen::VectorXd &in, Eigen::VectorXd &out,
			const double m, const double lambda,
			const double a_t, const double H_t, const double inv_ah_sqr,
			const long long int N)
{
  thrust::device_vector<double> x(2 * N*N*N);
  thrust::device_vector<double> dxdt(2 * N*N*N);
  thrust::copy(in.begin(), in.end(), x.begin());
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  //KG_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, a_t, H_t, inv_ah_sqr, N);
  Lambda_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, lambda, H_t, inv_ah_sqr, N);

  thrust::copy(dxdt.begin(), dxdt.end(), out.begin());
}


void kernel_test(const thrust::device_vector<double> &R_fft, thrust::device_vector<double> &Psi, thrust::device_vector<double> &dPsidt,
		 const long long int N, const double L, const double m,
		 const double a_t, const double H_t, const double eta_t, const double inv_ah_sqr,
		 const double t, fftWrapperDispatcher<thrust::device_vector<double>>::Generic &fft_wrapper)
{
  const double factor = (2 * std::numbers::pi / L) * eta_t / sqrt(3);
  std::cout << "factor = " << factor << '\n';
  {
    // thrust::device_vector<double> Psi_fft(R_fft.size());
    // thrust::device_vector<double> dPsidt_fft(R_fft.size());
    
    // dim3 threadsPerBlock_fft(8, 4, 4);
    // dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4/2+1);
    // Comoving_Curvature_Psi_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(R_fft.data()), thrust::raw_pointer_cast(Psi_fft.data()), N, factor);
    // Psi = fft_wrapper.execute_z2d(Psi_fft);


    dim3 threadsPerBlock_fft(8, 4, 4);
    dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4/2+1);
    Comoving_Curvature_Psi_dPsidt_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(R_fft.data()), thrust::raw_pointer_cast(Psi.data()), thrust::raw_pointer_cast(dPsidt.data()), N, factor, H_t);
    // Psi = fft_wrapper.execute_z2d(Psi_fft);
    // dPsidt = fft_wrapper.execute_z2d(dPsidt_fft);
    fft_wrapper.execute_inplace_z2d(Psi);
    fft_wrapper.execute_inplace_z2d(dPsidt);
  }
}


void CudaKleinGordonEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  KG_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, H_t, inv_ah_sqr, N);
}


CudaKleinGordonEquationInFRW::Vector CudaKleinGordonEquationInFRW::compute_energy_density(const Workspace &workspace, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double lambda = 0;
  const double a_t = workspace.cosmology.a(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  thrust::device_vector<double> rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(rho.data()), m, lambda, inv_ah_sqr, N);
  
  return rho;
}


CudaKleinGordonEquationInFRW::Vector CudaKleinGordonEquationInFRW::compute_dot_energy_density(const Workspace &workspace, const double t)
{
  const auto N = workspace.N;
  const auto L = workspace.L;
  const auto m = workspace.m;
  const auto a_t = workspace.cosmology.a(t);
  const auto H_t = workspace.cosmology.H(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  thrust::device_vector<double> dot_rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  dot_energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(dot_rho.data()), m, inv_ah_sqr, H_t, N);
  
  return dot_rho;
}


void CudaLambdaEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double lambda = workspace.lambda;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Lambda_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, lambda, H_t, inv_ah_sqr, N);
}


CudaLambdaEquationInFRW::Vector CudaLambdaEquationInFRW::compute_energy_density(const Workspace &workspace, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double lambda = workspace.lambda;
  const double a_t = workspace.cosmology.a(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  thrust::device_vector<double> rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(rho.data()), m, lambda, inv_ah_sqr, N);
  
  return rho;
}


void CudaSqrtPotentialEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double f_a = workspace.f_a;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Sqrt_Potential_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), m, f_a, H_t, inv_ah_sqr, N);
}


CudaSqrtPotentialEquationInFRW::Vector CudaSqrtPotentialEquationInFRW::compute_energy_density(const Workspace &workspace, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double f_a = workspace.f_a;
  const double a_t = workspace.cosmology.a(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  thrust::device_vector<double> rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Sqrt_Potential_energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(rho.data()), m, f_a, inv_ah_sqr, N);
  
  return rho;
}


void CudaFixedCurvatureEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const auto N = workspace.N;
  const auto L = workspace.L;
  const auto m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double H_t = workspace.cosmology.H(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Fixed_Curvature_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), thrust::raw_pointer_cast(workspace.Psi.data()), m, H_t, inv_ah_sqr, N);
}

// TODO
CudaFixedCurvatureEquationInFRW::Vector CudaFixedCurvatureEquationInFRW::compute_energy_density(const Workspace &workspace, const double t)
{
  const long long int N = workspace.N;
  const double L = workspace.L;
  const double m = workspace.m;
  const double a_t = workspace.cosmology.a(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);

  thrust::device_vector<double> rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Curvature_energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(rho.data()), thrust::raw_pointer_cast(workspace.Psi.data()), m, inv_ah_sqr, N);
  
  return rho;
}


void CudaComovingCurvatureEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const auto N = workspace.N;
  const auto L = workspace.L;
  const auto m = workspace.m;
  const auto a_t = workspace.cosmology.a(t);
  const auto H_t = workspace.cosmology.H(t);
  const auto eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  const double factor = (2 * std::numbers::pi / L) * eta_t / sqrt(3);
  
  // thrust::device_vector<double> Psi;
  // thrust::device_vector<double> dPsidt;

  // dim3 threadsPerBlock_fft(8, 4, 4);
  // dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4/2+1);
  // const double factor = (2 * std::numbers::pi / L) * eta_t / sqrt(3);
  // {
  //   thrust::device_vector<double> Psi_fft(workspace.R_fft.size());
  //   Comoving_Curvature_Psi_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.R_fft.data()), thrust::raw_pointer_cast(Psi_fft.data()), N, factor);
  //   Psi = workspace.fft_wrapper.execute_z2d(Psi_fft);
  // }
  // {
  //   thrust::device_vector<double> dPsidt_fft(workspace.R_fft.size());
  //   Comoving_Curvature_dPsidt_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.R_fft.data()), thrust::raw_pointer_cast(dPsidt_fft.data()), N, factor, H_t);
  //   dPsidt = workspace.fft_wrapper.execute_z2d(dPsidt_fft);
  // }

  thrust::device_vector<double> Psi(workspace.R_fft.size());
  thrust::device_vector<double> dPsidt(workspace.R_fft.size());
  
  dim3 threadsPerBlock_fft(8, 4, 4);
  dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4/2+1);
  Comoving_Curvature_Psi_dPsidt_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.R_fft.data()), thrust::raw_pointer_cast(Psi.data()), thrust::raw_pointer_cast(dPsidt.data()), N, factor, H_t);
  // Note: After Z2D transform, Psi and dPsidt are in FFTW padded format, and the index of grid point (a,b,c) is PADDED_IDX_OF(N,a,b,c) instead of IDX_OF(N,a,b,c).
  workspace.fft_wrapper.execute_inplace_z2d(Psi);
  workspace.fft_wrapper.execute_inplace_z2d(dPsidt);

  // thrust::device_vector<double> Psi;
  // thrust::device_vector<double> dPsidt;

  // thrust::device_vector<double> Psi_fft(workspace.R_fft.size());
  // thrust::device_vector<double> dPsidt_fft(workspace.R_fft.size());
  
  // dim3 threadsPerBlock_fft(8, 4, 4);
  // dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4/2+1);
  // Comoving_Curvature_Psi_dPsidt_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.R_fft.data()), thrust::raw_pointer_cast(Psi_fft.data()), thrust::raw_pointer_cast(dPsidt_fft.data()), N, factor, H_t);
  // Psi = workspace.fft_wrapper.execute_z2d(Psi_fft);
  // dPsidt = workspace.fft_wrapper.execute_z2d(dPsidt_fft);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Variable_Curvature_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), thrust::raw_pointer_cast(Psi.data()), thrust::raw_pointer_cast(dPsidt.data()), m, H_t, inv_ah_sqr, N);
}

CudaComovingCurvatureEquationInFRW::Vector CudaComovingCurvatureEquationInFRW::compute_energy_density(Workspace &workspace, const double t)
{
  const auto N = workspace.N;
  const auto L = workspace.L;
  const auto m = workspace.m;
  const auto a_t = workspace.cosmology.a(t);
  const auto eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  
  thrust::device_vector<double> Psi;
  
  dim3 threadsPerBlock_fft(8, 4, 4);
  dim3 numBlocks_fft((int)N/8, (int)N/4, (int)N/4);
  const double factor = (2 * std::numbers::pi / L) * eta_t / sqrt(3);
  {
    thrust::device_vector<double> Psi_fft(workspace.R_fft.size());
    Comoving_Curvature_Psi_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.R_fft.data()), thrust::raw_pointer_cast(Psi_fft.data()), N, factor);
    Psi = workspace.fft_wrapper.execute_z2d(Psi_fft);
  }
  
  thrust::device_vector<double> rho(N * N * N);
  
  dim3 threadsPerBlock((int)N, 1);
  dim3 numBlocks((int)N, (int)N);
  Curvature_energy_density_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(workspace.state.data()), thrust::raw_pointer_cast(rho.data()), thrust::raw_pointer_cast(Psi.data()), m, inv_ah_sqr, N);
  
  return rho;
}


// TODO
void CudaApproximateComovingCurvatureEquationInFRW::operator()(const State &x, State &dxdt, const double t)
{
  const auto N = workspace.N;
  const auto L = workspace.L;
  const auto m = workspace.m;
  const auto a_t = workspace.cosmology.a(t);
  const auto H_t = workspace.cosmology.H(t);
  const auto eta_t = workspace.cosmology.eta(t);
  const double inv_ah_sqr = 1.0 / ((L / N) * (L / N)) / (a_t * a_t);
  const double factor = (2 * std::numbers::pi / L) * eta_t / sqrt(3);

  const int M = workspace.M;
  //const double k_UV_M = (2 * std::numbers::pi / L) * (M / 2);
  //const double k_UV_eta = k_UV_M * eta_t;
  const bool use_approximation = true; //k_UV_eta >= 100.0;

  if(use_approximation) {
    // TODO
    // static thrust::device_vector<double> cutoff_R_fft = compute_cutoff_fouriers(N, M, workspace.R_fft);
    // static fftWrapperDispatcher<thrust::device_vector<double>>::Generic fft_wrapper_M(M);
    if(!workspace.Psi_approximation_initialized) {
      workspace.cutoff_R_fft = compute_cutoff_fouriers(N, M, workspace.R_fft);
      workspace.fft_wrapper_M_ptr = std::make_unique<typename fftWrapperDispatcher<thrust::device_vector<double>>::Generic>(M);
    }

    thrust::device_vector<double> Psi(workspace.cutoff_R_fft.size());
    thrust::device_vector<double> dPsidt(workspace.cutoff_R_fft.size());
  
    dim3 threadsPerBlock_fft(8, 4, 4);
    dim3 numBlocks_fft((int)M/8, (int)M/4, (int)M/4/2+1);
    Comoving_Curvature_Psi_dPsidt_fft_kernel<<<numBlocks_fft, threadsPerBlock_fft>>>(thrust::raw_pointer_cast(workspace.cutoff_R_fft.data()), thrust::raw_pointer_cast(Psi.data()), thrust::raw_pointer_cast(dPsidt.data()), M, factor, H_t);
    
    // Note: After Z2D transform, Psi and dPsidt are in FFTW padded format, and the index of grid point (a,b,c) is PADDED_IDX_OF(M,a,b,c) instead of IDX_OF(M,a,b,c).
    workspace.fft_wrapper_M_ptr->execute_inplace_z2d(Psi);
    workspace.fft_wrapper_M_ptr->execute_inplace_z2d(dPsidt);

    dim3 threadsPerBlock((int)N, 1);
    dim3 numBlocks((int)N, (int)N);
    Approximate_Variable_Curvature_FRW_equation_kernel<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(dxdt.data()), thrust::raw_pointer_cast(Psi.data()), thrust::raw_pointer_cast(dPsidt.data()), m, H_t, inv_ah_sqr, N, M);

  } else {
    CudaComovingCurvatureEquationInFRW full_eqn(workspace);
    full_eqn(x, dxdt, t);
  }
}



CudaApproximateComovingCurvatureEquationInFRW::Vector CudaApproximateComovingCurvatureEquationInFRW::compute_energy_density(Workspace &workspace, const double t)
{
  return CudaComovingCurvatureEquationInFRW::compute_energy_density(workspace, t);
}



// Explicit template instantiation definition for the thrust library.
template double thrust::reduce(const thrust::detail::execution_policy_base<thrust::cuda_cub::tag> &, thrust_const_iterator, thrust_const_iterator, double, boost::numeric::odeint::detail::maximum<double>);
