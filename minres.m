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
## reached.
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

function [x, flag, relres, iter, resvec]  = minres(A, b, tol, ...
  maxit, m1, m2, x0, varargin)
  ## Check the inputs
  if (nargin < 2)
      print_usage();
  endif

  [mb, nb] = size(b);
  if (nb != 1)
        print_usage();
  endif
  
  Aisnum = isnumeric(A);
  if Aisnum
    ## If A is a matrix
    [ma, na] = size(A);
    
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
    ## Preconditioner exists
    m1exist = true;
    misnum = isnumeric(m1);
      if (nargin >= 6) && !isempty(m2)
        m2exist = true;
      else
        m2exist = false;
      endif
  else
    ## Preconditioner does not exist
      m1exist = false;
      m2exist = false;
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
  flag = 1;

  n = length(b);
  beta = zeros(N, 1);
  alpha = zeros(N, 1);
  gamma = zeros(N, 1);
  delta = zeros(N, 1);
  epsilon = zeros(N, 1);
  c = zeros(N, 1);
  s = zeros(N, 1);
  resvec = zeros(N, 1);
  if Aisnum
      resvec(1) = norm(A * x0 - b);
  else
      resvec(1) = norm(feval(A, x0, varargin{:}) - b);
  endif

  if (isequal(b, zeros(n, 1)))
    ## If b is a zero vector
    x = zeros(mb, 1);
    flag = 0;
    relres = NaN;
    iter = 0;
    resvec = [resvec(1); 0];
    return
  endif
  
  if m1exist
    if m2exist
      if misnum
        by = m2 \ (m1 \ b);
      else
        by = feval(m2, feval(m1, b, varargin{:}), varargin{:});
      endif
    else
      if misnum
        by = m1 \ b;
      else
        by = feval(m1, b, varargin{:});
      endif
    endif
  else
    by = b;
  endif

  ## Initiation
  v_0 = zeros(n, 1);
  if Aisnum
    if m1exist
      if m2exist
        if misnum
          b0 = by - m2 \ (m1 \ (A * x0));
        else
          b0 = by - feval(m2, feval(m1, A * x0, varargin{:}), varargin{:});
        endif
      else
        if misnum
          b0 = by - m1 \ (A * x0);
        else
          b0 = by - feval(m1, A * x0, varargin{:});
        endif
      endif
    else
      b0 = by - A * x0;
    endif
  else
    if m1exist
      if m2exist
        if misnum
          b0 = by - m2 \ (m1 \ feval(A, x0, varargin{:}));
        else
          b0 = by - feval(m2, feval(m1, feval(A, x0, varargin{:})),...
          varargin{:});
        endif
      else
        if misnum
          b0 = by - m1 \ feval(A, x0, varargin{:});
        else
          b0 = by - feval(m1, feval(A, x0, varargin{:}), varargin{:});
        endif
      endif
    else
      b0 = by - feval(A, x0, varargin{:});
    endif
  endif
  beta(1) = norm(b0);
  relres = beta(1) / norm(by);
  if (relres <= tol) || (beta(1) <= eps)
    ## If x0 is already good enouph
    x = x0;
    flag = 0;
    iter = 0;
    resvec = [resvec(1)];
    return
  endif
  v_1 = b0 / beta(1);
        
  v_p = v_0;
  v_n = v_1;
  temp2 = beta(1);
  m_p = zeros(n, 1);
  m_pp = zeros(n, 1);
  x = x0;

  ## Iteration
  for k = 1: (N - 1)
    if Aisnum
      if m1exist
        if m2exist
          if misnum
            temp0 = m2 \ (m1 \ (A * v_n));
          else
            temp0 = feval(m2, feval(m1, A * v_n, varargin{:}), varargin{:});
          endif
        else
          if misnum
            temp0 = m1 \ (A * v_n);
          else
            temp0 = feval(m1, A * v_n, varargin{:});
          endif
        endif
      else
        temp0 = A * v_n;
      endif
    else
      if m1exist
        if m2exist
          if misnum
            temp0 = m2 \ (m1 \ feval(A, v_n, varargin{:}));
          else
            temp0 = feval(m2, feval(m1, feval(A, v_n, varargin{:}),...
            varargin{:}), varargin{:});
          endif
        else
          if misnum
            temp0 = m1 \ feval(A, v_n, varargin{:});
          else
            temp0 = feval(m1, feval(A, v_n, varargin{:}), varargin{:});
          endif
        endif
      else
        temp0 = feval(A, v_n, varargin{:});
      endif
    endif
    alpha(k) = v_n' * temp0;
    temp1 = temp0 - alpha(k) * v_n - beta(k) * v_p;
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

    m = 1 / gamma(k) * (v_n - epsilon(k) * m_pp - delta(k) * m_p);

    x = x + m * temp2 * c(k);
    temp2 = temp2 * s(k);

    
    if Aisnum
       r = norm(A * x - b);
    else
       r = norm(feval(A, x, varargin{:}) - b);
    endif
    relres = r / norm(b);
    resvec(k + 1) = r;
    iter = k;
    ## Check convergence
    if (relres <= tol) || (beta(k + 1) <= eps)
       flag = 0;
       resvec = resvec(1: (k + 1));
       break
    endif
    v_f = temp1 / beta(k + 1);

    m_pp = m_p;
    m_p = m;
    v_p = v_n;
    v_n = v_f;

  endfor 
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
%!

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
%! assert (norm (A * x - b) / norm (b), 0, 1e-6);
%! assert (flag, 0);

%!test
%! ## solve small indefinite hermitian system
%!
%! B = diag([0;1;-2])
%! U = [1/sqrt(2), 1/sqrt(2), 0;
%!   -1/sqrt(2)*i, 1/sqrt(2)*i,0;
%!     0,0,i];
%! A = U * B * U'; 
%! b = sum(A, 2);
%! [x, flag] = minres (A, b, [], 3);
%! assert (norm (A * x - b) / norm (b), 0, 1e-6);
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
%! ## matlab minres converged at iteration 40, but minres here needs more
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