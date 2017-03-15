## Copyright (C) 2017 Xie Rui
##
## This file is part of Octave.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## Usage
##
## @itemize
## @item
## x = minres(A, b)
## @item
## minres (A, b, tol)
## @item
## minres (A, b, tol, maxit)
## @item
## minres (A, b, [], maxit)
## @item
## minres (A, b, tol, maxit, m)
## @item
## minres (A, b, [], [], m)
## @item
## minres (A, b, tol, maxit, m1, m2)
## @item
## minres (A, b, tol, maxit, m1, m2, x0)
## @item
## minres (A, b, tol, maxit, m, [], x0)
## @item
## [x, flag] = minres (A, b, @dots{})
## @item
## [x, flag, relres] = minres (A, b, @dots{})
## @item
## [x, flag, relres, iter] = minres (A, b, @dots{})
## @item
## [x, flag, relres, iter, resvec] = minres (A, b, @dots{})
## @item
## [x, flag, relres, iter, resvec, resveccg] = minres (A, b, @dots{})
## @end itemize
##
##
## Solve the linear system of equations @w{@code{@var{A} * @var{x} = @var{b}}}
## by means of the Minimum Residual Method.
##
##
## The input arguments are
##
## @itemize
## @item
## @var{A} should be a square and symmetric (preferably sparse) matrix 
## which may be both indefinite and singular or a function
## handle, inline function or string containing the name of a function
## which computes @w{@code{@var{A} * @var{x}}}.
##
## @item
## @var{b} is the right-hand side vector.
##
## @item
## @var{tol} is the required relative tolerance for the residual error,
## @w{@code{@var{b} - @var{A} * @var{x}}}.  
## If @var{tol} is omitted or empty then a tolerance of 1e-6 is used.
##
## @item
## @var{maxit} is the maximum allowable number of iterations; if @var{maxit}
## is omitted or empty then a value of 100 is used.
##
## @item
## @var{m} = @var{m1} * @var{m2} is the preconditioning matrix, so that
## the iteration is (theoretically) equivalent to solving by @code{minres}
## @w{@code{@var{P} * @var{x} = @var{m} \ @var{b}}}, with
## @w{@code{@var{P} = @var{m} \ @var{A}}}. Instead of matrices @var{m1} and
## @var{m2}, the user may pass two functions which return the results of
## applying the inverse of @var{m1} and @var{m2} to a vector.
## If @var{m1} is omitted or empty @code{[]} then no preconditioning is applied.
## If @var{m2} is omitted, @var{m} = @var{m1} will be used as a preconditioner.
##
## @item
## @var{x0} is the initial guess.  If @var{x0} is omitted or empty then the
## function sets @var{x0} to a zero vector by default.
## @end itemize
##
## The output arguments are
##
## @itemize
## @item
## @var{x} is the computed approximation to the solution of
## @w{@code{@var{A} * @var{x} = @var{b}}}.
##
## @item
## @var{flag} reports on the convergence.  A value of 0 means the solution
## converged and the tolerance criterion given by @var{tol} is satisfied.
## A value of 1 means that the @var{maxit} limit for the iteration count was
## reached. A value of 2 means that M is ill-conditioned.
## A value of 3 means that minres stagnated. (Two consecutive iterates
## were the same.) A value of 4 means that A is not hermitian.
##
## @item
## @var{relres} is the final relative residual,
## measured in the Euclidean norm.
##
## @item
## @var{iter} is the actual number of iterations performed.
##
## @item
## @var{resvec(i)} is the Euclidean norm of the residual after the
## (@var{i}-1)-th iteration, @code{@var{i} = 1, 2, @dots{}, @var{iter}+1}.
##
## @item
## @var{resveccg(i)} is the Euclidean norm of the Conjugate Gradients residual 
## after the (@var{i}-1)-th iteration, 
## @code{@var{i} = 1, 2, @dots{}, @var{iter}+1}.
##
## @end itemize
##
##
## Let us consider a trivial problem with a diagonal matrix (we exploit the
## sparsity of A)
##
## @example
## @group
## n = 10;
## A = diag (sparse (1:n));
## b = rand (n, 1);
## [l, u, p] = ilu (A, struct ("droptol", 1.e-3));
## @end group
## @end example
##
## @sc{Example 1:} Simplest use of @code{minres}
##
## @example
## x = minres (A, b)
## @end example
##
## @sc{Example 2:} @code{minres} with a function which computes
## @code{@var{A} * @var{x}}
##
## @example
## @group
## function y = apply_a (x)
##   y = [1: 10]' .* x;
## endfunction
##
## x = minres ("apply_a", b)
## @end group
## @end example
##
## @sc{Example 3:} @code{minres} with a preconditioner: @var{l} * @var{u}
##
## @example
## x = minres (A, b, 1.e-6, 500, l*u)
## @end example
##
## @sc{Example 4:} @code{minres} with a preconditioner: @var{l} * @var{u}.
## Faster than @sc{Example 3} since lower and upper triangular matrices are
## easier to invert
##
## @example
## x = minres (A, b, 1.e-6, 500, l, u)
## @end example
##
## @sc{Example 5:} @code{minres} when @var{A} is indefinite. It fails 
## with @code{pcg}.
##
## @example
## A = diag([20:-1:1, -1:-1:-20]);
## b = sum(A,2);
## x = minres(A, b)
## @end example
##
## Reference:
##
## @enumerate
## @item
## C. C. PAIGE and M. A. SAUNDERS, @cite{Solution of Sparse Indefinite 
## Systems of Linear Equations},
## SIAM J. Numer. Anal., 1975. (the minimum residual method)
##
## @end enumerate

function [x, flag, relres, iter, resvec, resveccg]  = minres(A, b, tol, ...
  maxit, m1, m2, x0, varargin)
  ## Check the inputs
  if (nargin < 2)
      print_usage();
  endif

  [mb, nb] = size(b);
  if (nb != 1)
        print_usage();
  endif
  
  flag = 1;
  Aisnum = isnumeric(A);
  if Aisnum
    [ma, na] = size(A);
    if !ishermitian(A, 1e-6)
      flag = 4;
    endif
    if (ma != na)
        print_usage();
    endif
    if (na != mb)
        print_usage();
    endif
  endif

  if (nargin < 3) || isempty(tol)
    tol = 1e-6;
  endif
    
  if (nargin < 4) || isempty(maxit)
    maxit = min(100, mb + 5);
  endif

  if (nargin >= 5) && !isempty(m1)
    m1exist = true;
    m1isnum = isnumeric(m1);
      if (nargin >= 6) && !isempty(m2)
        m2exist = true;
        m2isnum = isnumeric(m2);
      else
        m2exist = false;
        m2isnum = false;
      endif
  else
      m1exist = false;
      m2exist = false;
      m1isnum = false;
      m2isnum = false;
  endif

  if (nargin >= 7) && !isempty(x0)
      [mx0, nx0] = size(x0);
      if (mx0 != mb) || (nx0 != 1)
          print_usage();
      endif
  else
      x0 = zeros(mb, 1);
  endif

        
  ## Preallocation
  N = maxit + 1;

  n = length(b);
  beta = zeros(N, 1);
  alpha = zeros(N, 1);
  gamma = zeros(N, 1);
  delta = zeros(N, 1);
  epsilon = zeros(N, 1);
  c = zeros(N, 1);
  s = zeros(N, 1);
  resvec = zeros(N, 1);
  resveccg = zeros(N, 1);
  if Aisnum
      resvec(1) = norm(A * x0 - b);
  else
      resvec(1) = norm(feval(A, x0, varargin{:}) - b);
  endif
  resveccg(1) = resvec(1);

  if (isequal(b, zeros(n, 1)))
    ## If b is a zero vector
    x = zeros(mb, 1);
    if !(flag == 4)
      flag = 0;
    endif
    relres = NaN;
    iter = 0;
    resvec = [resvec(1); 0];
    resveccg = resvec;
    return
  endif

  ## Initiation
  if Aisnum
    rM = b - A * x0;
  else
    rM = b - feval(A, x0, varargin{:});
  endif
  if m1exist
    try
      warning("error","Octave:singular-matrix","local");
      if m1isnum
        b0 = m1 \ rM;
      else
        b0 = feval(m1, rM, varargin{:});
      endif
      if m2exist
        if m2isnum
          b0 = m2 \ b0;
        else
          b0 = feval(m2, b0, varargin{:});
        endif
      endif
    catch
      flag = 2;
    end_try_catch
  else
    b0 = rM;
  endif
  
  if (flag == 2)||(m1exist && (!all(isfinite(b0))))
    ## M is ill-conditioned
    x = x0;
    iter = 0;
    resvec = resvec(1);
    resveccg = resveccg(1);
    return
  endif
  
  beta(1) = norm(b0);
  normb = norm(b);
  relres = resvec(1) / normb;
  x = x0;
  
  if (relres <= tol) || (beta(1) <= eps)
    ## If x0 is already good enouph
    if !(flag == 4)
      flag = 0;
    endif
    iter = 0;
    resvec = resvec(1);
    resveccg = resveccg(1);
    return
  endif
        
  v_p = zeros(n, 1);
  v_n = b0 / beta(1);
  temp2 = beta(1);
  m_p = zeros(n, 1);
  m_pp = zeros(n, 1);
  Am_p = zeros(n, 1);
  Am_pp = zeros(n, 1);
  x = x0;

  ## Iteration
  for k = 1: (N - 1)
    if Aisnum
      Av_n = A * v_n;
    else
      Av_n = feval(A, v_n, varargin{:});
    endif
    if m1exist
      try
        warning("error","Octave:singular-matrix","local");
        if m1isnum
          MAv_n = m1 \ Av_n;
        else
          MAv_n = feval(m1, Av_n, varargin{:});
        endif
        if m2exist
          if m2isnum
            MAv_n = m2 \ MAv_n;
          else
            MAv_n = feval(m2, MAv_n, varargin{:});
          endif
        endif
      catch
        flag = 2;
      end_try_catch
    else
      MAv_n = Av_n;
    endif
    
    if (flag == 2)||(m1exist && (!all(isfinite(MAv_n))))
      flag = 2;
      iter = k - 1;
      resvec = resvec(1: k);
      resvec = resveccg(1: k);
      return
    endif
    
    alpha(k) = v_n' * MAv_n;
    temp1 = MAv_n - alpha(k) * v_n - beta(k) * v_p;
    beta(k + 1) = norm(temp1);

    if k > 2
       epsilon(k) = s(k - 2) * beta(k);
       gamma_h = - c(k - 2) * beta(k) * s(k - 1) - alpha(k) * c(k - 1);
       delta(k) = - c(k - 2) * beta(k) * c(k - 1) + alpha(k) * s(k - 1);
    elseif k == 2
       epsilon(k) = 0;
       gamma_h = beta(2) * s(1) - alpha(2) * c(1);
       delta(k) = beta(2) * c(1) + alpha(2) * s(1);
    else
       epsilon(k) = 0;
       gamma_h = alpha(1);
       delta(k) = 0;
    endif

    gamma(k) = sqrt(gamma_h ^ 2 + beta(k + 1) ^ 2);
    c(k) = gamma_h / gamma(k);
    s(k) = beta(k + 1) / gamma(k);

    m = (v_n - epsilon(k) * m_pp - delta(k) * m_p) / gamma(k);
    Am = (Av_n - epsilon(k) * Am_pp - delta(k) * Am_p) / gamma(k);

    x = x + m * temp2 * c(k);
    rM = rM - Am * temp2 * c(k);
    
    normrM = norm(rM);
    temp2 = temp2 * s(k);
    
    rcg = rM - temp2 * s(k) / c(k) * Am;
    
    relresn = relres;
    relres = normrM / normb;
    resvec(k + 1) = normrM;
    resveccg(k + 1) = norm(rcg);
    iter = k;
    ## Check convergence
    if (relres <= tol) || (beta(k + 1) <= eps)
      if !(flag == 4)
       flag = 0;
      endif
      resvec = resvec(1: (k + 1));
      resveccg = resveccg(1: (k + 1));
      break
    endif
    
    if (resvec(k + 1) == resvec(k))&&!(flag == 4)
      flag = 3;
    elseif (flag == 3)
      flag = 1;
    endif

    m_pp = m_p;
    m_p = m;
    Am_pp = Am_p;
    Am_p = Am;
    v_p = v_n;
    v_n = temp1 / beta(k + 1);
    
  endfor 
  if (iter == 1)
    warning ("iteration converged too fast");
  endif
endfunction


%!demo
%! ## Simplest usage of minres (see also 'help minres')
%! ## Note that A is indefinite
%!
%! A = diag([20:-1:1, -1:-1:-20]);
%! b = sum(A,2);
%! y = A \ b;  # y is the true solution
%! x = minres (A, b);
%! printf ("The solution relative error is %g\n", norm (x - y) / norm (y));

%!demo # simplest use
%!
%! n = 10;
%! A = toeplitz (sparse ([1, 1], [1, 2], [2, 1], 1, n));
%! b = A * ones (n, 1);
%! M1 = ichol (A); # in this tridiagonal case it corresponds to chol (A)'
%! M2 = M1';
%! M = M1 * M2;
%! x = minres (A, b);
%! Afun = @(x) A * x;
%! x = minres (Afun, b);
%! x = minres (A, b, 1e-6, 100, M);
%! x = minres (A, b, 1e-6, 100, M1, M2);
%! Mfun = @(x) M \ x;
%! x = minres (Afun, b, 1e-6, 100, Mfun);
%! M1fun = @(x) M1 \ x;
%! M2fun = @(x) M2 \ x;
%! x = minres (Afun, b, 1e-6, 100, M1fun, M2fun);
%! function y = Ap (A, x, p) # compute A^p * x
%!    y = x;
%!    for i = 1:p
%!      y = A * y;
%!    endfor
%!  endfunction
%! Afun = @(x, p) Ap (A, x, p);
%! x = minres (Afun, b, [], [], [], [], [], 2); # solution of A^2 * x = b

%!demo
%!
%! n = 10;
%! A = toeplitz (sparse ([1, 1], [1, 2], [2, 1], 1, n));
%! b = A * ones (n, 1);
%! M1 = ichol (A + 0.1 * eye (n)); # factorization of A perturbed
%! M2 = M1';
%! M = M1 * M2;
%!
%! ## reference solution computed by pcg after two iterations
%! [x_ref, fl] = minres (A, b, [], 2, M);
%! x_ref
%!
%! ## split preconditioning
%! [y, fl] = minres ((M1 \ A) / M2, M1 \ b, [], 2);
%! x = M2 \ y # compare x and x_ref

%!demo
%! ## Full output from minres
%! ## We use this output to plot the convergence history
%!
%! N = 10;
%! A = diag ([1:N]); b = rand (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag, relres, iter, resvec] = minres (A, b);
%! printf ("The solution relative error is %g\n", norm (x - X) / norm (X));
%! title ("Convergence history");
%! semilogy ([0:iter], resvec / resvec(1), "o-g");
%! xlabel ("Iteration"); ylabel ("log(||b-Ax||/||b||)");
%! legend ("relative residual");


%!test
%! ## solve small diagonal system
%!
%! N = 10;
%! A = diag ([1:N]); b = rand (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag] = minres (A, b, [], N+1);
%! assert (norm (x - X) / norm (X), 0, 1e-10);
%! assert (flag, 0);

%!test
%! ## solve small indefinite diagonal system
%!
%! N = 10;
%! A = diag([1:N] .* (-ones(1, N) .^ 2)); b = rand (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag] = minres (A, b, [], N+1);
%! assert (norm (x - X) / norm (X), 0, 1e-10);
%! assert (flag, 0);

%!test
%! ## solve small singular diagonal system
%!
%! N = 10;
%! A = diag([(-1):(N - 2)]); 
%! b = sum(A, 2);
%! [x, flag] = minres (A, b, [], N+1);
%! assert (norm (A * x - b) / norm (b), 0, 1e-10);
%! assert (flag, 0);

%!test
%! ## solve small indefinite hermitian system
%!
%! B = diag([0;1;-2]);
%! U = [1/sqrt(2), 1/sqrt(2), 0;
%!   -1/sqrt(2)*i, 1/sqrt(2)*i,0;
%!     0,0,i];
%! A = U * B * U'; 
%! b = sum(A, 2);
%! [x, flag] = minres (A, b, [], 3);
%! assert (norm (A * x - b) / norm (b), 0, 1e-10);
%! assert (flag, 0);

%!test
%! ## solve tridiagonal system, do not converge in 20 iterations
%!
%! N = 100;
%! A = zeros (N, N);
%! for i = 1 : N - 1 # form 1-D Laplacian matrix
%!   A(i:i+1, i:i+1) = [2 -1; -1 2];
%! endfor
%! b = ones (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag, relres, iter, resvec] = minres (A, b, 1e-12, 20);
%! assert (flag);
%! assert (relres > 0.1);
%! assert (iter, 20); # should perform max allowable default number of iterations

%!warning <iteration converged too fast>
%! ## solve tridiagonal system with "perfect" preconditioner which converges
%! ## in one iteration
%!
%! N = 100;
%! A = zeros (N, N);
%! for i = 1 : N - 1  # form 1-D Laplacian matrix
%!   A(i:i+1, i:i+1) = [2 -1; -1 2];
%! endfor
%! b = ones (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag, relres, iter, resvec] = minres (A, b, [], [], A, [], b);
%! assert (norm (x - X) / norm (X), 0, 1e-6);
%! assert (flag, 0);
%! assert (iter, 1);  # should converge in one iteration

%!test
%! ## test for algorithm accuracy and compatibility from matlab doc example
%!
%! n = 100;
%! on = ones (n, 1); 
%! A = spdiags ([-2*on 4*on -2*on], -1:1, n, n);
%! b = sum (A, 2); 
%! tol = 1e-10; 
%! maxit = 50; 
%! M1 = spdiags (4*on, 0, n, n);
%! x = minres (A, b, tol, maxit, M1);
%! assert (size (x), [100, 1]);
%! assert (x,ones(100,1),1e-13);

%!test
%! ## solve indefinite diagonal system
%! ## test for algorithm convergence rate from matlab doc example
%! ## matlab minres converged at iteration 40, so did minres here
%! ## pcg fails with this test
%!
%! A = diag([20:-1:1, -1:-1:-20]);
%! b = sum(A,2);  
%! tol = 1e-6; 
%! maxit = 45; 
%! [x, flag, relres, iter, resvec] = minres (A, b, tol, maxit);
%! assert (flag, 0);
%! assert (iter > 40);   
%! assert (size (x), [40, 1]);
%! assert (x,ones(40,1),1e-7);

%!test
%! ## solve indefinite hermitian system
%!
%! B = spdiags([50:-1:1, -1:-1:-50]', 0, 100, 100);
%! on = ones(100, 1);
%! P = spdiags([-i * on, on, i * on], [-1, 0, 1], 100, 100);
%! [U,R] = qr(P);
%! A = U * B * U';
%! A = (A + A') / 2;
%! b = sum(A,2);  
%! tol = 1e-10; 
%! maxit = 150; 
%! [x, flag, relres, iter, resvec] = minres (A, b, tol, maxit);
%! assert (flag, 0);
%! assert (size (x), [100, 1]);
%! assert (x,ones(100,1),1e-9);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## Check that all the subscripts works
%!
%! A = toeplitz (sparse ([2, 1 ,0, 0, 0]));
%! b = A * ones (5, 1);
%! M1 = diag (sqrt (diag (A)));
%! M2 = M1; # M1 * M2 is the Jacobi preconditioner
%! Afun = @(z) A*z;
%! M1_fun = @(z) M1 \ z;
%! M2_fun = @(z) M2 \ z;
%! [x, flag, ~, iter] = minres (A,b);
%! assert(flag, 0);
%! [x, flag, ~ , iter] = minres (A, b, [], [], M1 * M2);
%! assert(flag, 0);
%! [x, flag, ~ , iter] = minres (A, b, [], [], M1, M2);
%! assert(flag, 0);
%! [x, flag] = minres (A, b, [], [], M1_fun, M2_fun);
%! assert(flag, 0);
%! [x, flag] = minres (A, b,[],[], M1_fun, M2);
%! assert(flag, 0);
%! [x, flag] = minres (A, b,[],[], M1, M2_fun);
%! assert(flag, 0);
%! [x, flag] = minres (Afun, b);
%! assert(flag, 0);
%! [x, flag] = minres (Afun, b,[],[], M1 * M2);
%! assert(flag, 0);
%! [x, flag] = minres (Afun, b,[],[], M1, M2);
%! assert(flag, 0);
%! [x, flag] = pcg (Afun, b,[],[], M1_fun, M2);
%! assert(flag, 0);
%! [x, flag] = pcg (Afun, b,[],[], M1, M2_fun);
%! assert(flag, 0);
%! [x, flag] = minres (Afun, b,[],[], M1_fun, M2_fun);
%! assert(flag, 0);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## solve small diagonal system
%!
%! N = 10;
%! A = diag ([1:N]); b = rand (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag] = minres (A, b, [], N+1);
%! assert (norm (x - X) / norm (X), 0, 1e-10);
%! assert (flag, 0);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## A not positive definite
%!
%! N = 10;
%! A = -diag([1:N]); b = rand (N, 1);
%! X = A \ b;  # X is the true solution
%! [x, flag] = minres (A, b, [], N+1);
%! assert (flag, 0);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## A is a non-Hermitian matrix
%! ## minres recognized the wrong type of matrix
%!
%! N = 10;
%! A = diag (1:N) + 1i*1e-04*rand (N);
%! b = ones (N, 1);
%! [x,flag] = minres (A, b, []);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## A has a small imaginary part
%!
%! N = 10;
%! A = diag (1:N) + 1i*1e-10*rand (N);
%! b = ones (N, 1);
%! [x,flag] = minres (A, b, [], N+1);
%! assert (flag, 0);
%! assert (x, A\b, -1e-6);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## minres solves linear system with A Hermitian positive definite
%!
%! N = 20;
%! A = 2*rand (N)-1 + 1i*(2*rand (N)-1);
%! A = A'*A;
%! b = A * ones (N,1);
%! Hermitian_A = ishermitian (A);
%! [x,flag] = minres (A, b, [], 2*N);
%! assert (Hermitian_A, true)
%! assert (flag, 0);
%! assert (x, ones (N, 1), -1e-4);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## minres solves preconditioned linear system with A HPD
%!
%! N = 20;
%! A = 2*rand (N)-1 + 1i*(2*rand (N)-1);
%! A = A' * A;
%! b = A * ones (N,1);
%! M2 = chol (A + 0.1 * eye (N)); # factor of a perturbed matrix
%! M = M2' * M2;
%! Hermitian_A = ishermitian (A);
%! Hermitian_M = ishermitian (M);
%! [x,flag] = minres (A, b, [], 2*N, M);
%! assert (Hermitian_A, true);
%! assert (Hermitian_M, true);
%! assert (flag, 0);
%! assert (x, ones (N, 1), -1e-4);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## minres recognizes that the preconditioner matrix is singular
%!
%! N = 3;
%! A = rand(3);
%! A = A*A';
%! M = [1 0 0; 0 1 0; 0 0 0]; # the last rows is zero
%! [x,flag] = minres (A, ones(3,1), [], [], M);
%! assert (flag, 2);

%!test
%! ## modified test from pcg by Piotr Krzyzanowski, Vittoria Rezzonico
%! ## and Cristiano Dorigo
%! ## b is zero vector, so minres returns a zero vector
%!
%! A = rand (4);
%! A = A' * A;
%! [x, flag] = minres (A, zeros (4, 1), [], [], [], [], ones (4, 1));
%! assert (x, zeros (4, 1))