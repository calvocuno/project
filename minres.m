## -*- texinfo -*-
## @deftypefn  {} {@var{x} =} minres (@var{A}, @var{b}, @var{tol}, @var{maxit}, @var{m1}, @var{m2}, @var{x0})
## @deftypefnx {} {[@var{x}, @var{flag}, @var{relres}, @var{iter}, @var{resvec}] =} minres (@dots{})
##
## Solve the linear system of equations @w{@code{@var{A} * @var{x} = @var{b}}}
## by means of the Minimum Residual Method.
##
## The input arguments are
##
## @itemize
## @item
## @var{A} should be a square, real and symmetric (preferably sparse) matrix 
## which may be both indefinite and singular or a function
## handle, inline function or string containing the name of a function which
## computes @w{@code{@var{A} * @var{x}}}.
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
## @w{@code{@var{P} = @var{m} \ @var{A}}}.  
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
## @sc{Example 5:} @code{minres} when @var{A} is indefinite. It fails with @code{pcg}.
##
## @example
## A = diag([20:-1:1, -1:-1:-20]);
## b = sum(A,2);
## x = minres(A, b)
## @end example
##
## References:
##
## @enumerate
## @item
## C. C. PAIGE and M. A. SAUNDERS, @cite{Solution of Sparse Indefinite Systems of Linear Equations},
## SIAM J. Numer. Anal., 1975. (the minimum residual method)
##
## @end enumerate
## @end deftypefn

function [x, flag, relres, iter, resvec]  = minres(A, b, tol, maxit, m1, m2, x0, varargin)
  ## Check the inputs
  if (nargin < 2)
      error('Not enough inputs!');
  endif

  [mb, nb] = size(b);
  if (nb ~= 1)
        error('b is not a vector!');
  endif
  
  Aisnum = isnumeric(A);
  if Aisnum
    ## If A is a matrix
    [ma, na] = size(A);
    
    if (ma ~= na)
        error('A is not a square matrix!');
    endif
    if (na ~= mb)
        error('A and b are not matched!');
    endif
    if ~isequal(A, A')
        error('A is not symmetric!');
    endif
  endif

  if (nargin < 3) || isempty(tol)
    tol = 1e-6;
  endif
    
  if (nargin < 4) || isempty(maxit)
    maxit = min(100, mb + 5);
  endif

  if (nargin >= 5) && ~isempty(m1)
    ## Preconditioner exists
      if (nargin >= 6) && ~isempty(m2)
          M = m1 * m2;
          [mm, mn] = size(M);
          if (mm ~= mb) || (mn ~= mb)
            error('M has the wrong size!');
          endif
          Pinv = inv(m2) * inv(m1);
      else
          M = m1;
          [mm, mn] = size(M);
          if (mm ~= mb) || (mn ~= mb)
            error('M has the wrong size!');
          endif
          Pinv = inv(M);
      endif
  else
    ## Preconditioner does not exist
      Pinv = speye(mb);
  endif

  if (nargin >= 7) && ~isempty(x0)
      [mx0, nx0] = size(x0);
      if (mx0 ~= mb) || (nx0 ~= 1)
          error('x0 has the wrong size!');
      endif
  else
      x0 = zeros(mb, 1);
  endif

        
  ## Preallocation
  N = maxit;
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

  by = Pinv * b;

  ## Initiation
  v_0 = zeros(n, 1);
  if Aisnum
    b0 = by - Pinv * (A * x0);
  else
    b0 = by - Pinv * feval(A, x0, varargin{:});
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
  v(:, 1) = v_1; 
        
  v_p = v_0;
  v_n = v_1;
  temp2 = beta(1);
  m_p = zeros(n, 1);
  m_pp = zeros(n, 1);
  y = zeros(n, 1);

  ## Iteration
  for k = 1: (N - 1)
    if Aisnum
       alpha(k) = v_n' * (Pinv * (A * (Pinv * v_n)));
       temp1 = (Pinv * (A * (Pinv * v_n))) - alpha(k) * v_n - beta(k) * v_p;
    else
       alpha(k) = v_n' * (Pinv * feval(A, Pinv * v_n, varargin{:}));
       temp1 = (Pinv * feval(A, Pinv * v_n, varargin{:})) - alpha(k) * v_n - beta(k) * v_p;
    endif
    beta(k + 1) = norm(temp1);
    v_f = temp1 / beta(k + 1);
    v(:, k + 1) = v_f;

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

    y = y + m * temp2 * c(k);
    temp2 = temp2 * s(k);

    x = Pinv * y + x0;
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

    m_pp = m_p;
    m_p = m;
    v_p = v_n;
    v_n = v_f;

  endfor 
endfunction