function [x, flag, relres, iter, resvec]  = minres_3_8_1(A, b, tol, maxit, m1, m2, x0)
% MINRES Minimum Residual Method
%%
if (nargin < 2)
    error(message('Not enough inputs!'));
end

[ma, na] = size(A);
[mb, nb] = size(b);
if (ma ~= na)
    error(message('A is not a square matrix!'));
end
if (nb ~= 1)
    error(message('b is not a vector!'));
end
if (na ~= mb)
    error(message('A and b are not matched!'));
end
if ~isequal(A, A')
    error(message('A is not symmetric!'));
end

if (nargin < 3) || isempty(tol)
    tol = 1e-6;
end

if (nargin < 4) || isempty(maxit)
    maxit = min(100,na);
end

if (nargin >= 5) && ~isempty(m1)
    if (nargin >= 6) && ~isempty(m2)
        M = m1 * m2;
    else
        M = m1;
    end
    [mm, mn] = size(M);
    if (mm ~= na) || (mn ~= na)
        error(message('M has the wrong size!'));
    end
    preconditionerflag = true;
    P = sqrt(M);
    Pinv = inv(P);
else
    preconditionerflag = false;
    P = speye(na);
    Pinv = speye(na);
end

if (nargin >= 7) && ~isempty(x0)
    [mx0, nx0] = size(x0);
    if (mx0 ~= na) || (nx0 ~= 1)
        error(message('x0 has the wrong size!'));
    end
else
    x0 = zeros(na, 1);
end

    
%%
N = maxit;
flag = 1;
%%
n = length(b);
beta = zeros(N, 1);
alpha = zeros(N, 1);
gamma = zeros(N, 1);
delta = zeros(N, 1);
epsilon = zeros(N, 1);
c = zeros(N, 1);
s = zeros(N, 1);

if (preconditionerflag == 0)
    resvec = zeros(N, 1);
    resvec(1) = norm(A * x0 - b);
    %%
    v_0 = zeros(n, 1);
    b0 = b - A * x0;
    beta(1) = norm(b0);
    if beta(1) < tol
        x = x0;
        flag = 0;
        relres = NaN;
        iter = 0;
        return
    end
    v_1 = b0 / beta(1);
    v(:, 1) = v_1; 
    %%
    v_p = v_0;
    v_n = v_1;
    temp2 = beta(1);
    m_p = zeros(n, 1);
    m_pp = zeros(n, 1);
    x = x0;

    for k = 1: (N - 1)
        alpha(k) = v_n' * A * v_n;
        temp1 = A * v_n - alpha(k) * v_n - beta(k) * v_p;
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
        end

        gamma(k) = sqrt(gamma_h ^ 2 + beta(k + 1) ^ 2);
        c(k) = gamma_h / gamma(k);
        s(k) = beta(k + 1) / gamma(k);

        m = 1 / gamma(k) * (v_n - epsilon(k) * m_pp - delta(k) * m_p);

        x = x + m * temp2 * c(k);
        temp2 = temp2 * s(k);

        r = norm(A * x - b);
        relres = r / norm(b);
        resvec(k + 1) = r;
        iter = k;
        if (r < tol) || (beta(k + 1) < tol)
            flag = 0;
            resvec = resvec(1: (k + 1));
            break
        end

        m_pp = m_p;
        m_p = m;
        v_p = v_n;
        v_n = v_f;

    end  
else
    by = Pinv * b;
    y0 = P * x0;
    resvec = zeros(N, 1);
    resvec(1) = norm(A * x0 - b);
    %%
    v_0 = zeros(n, 1);
    b0 = by - Pinv * (A * x0);
    beta(1) = norm(b0);
    if beta(1) < tol
        x = x0;
        flag = 0;
        relres = NaN;
        iter = 0;
        return
    end
    v_1 = b0 / beta(1);
    v(:, 1) = v_1; 
    %%
    v_p = v_0;
    v_n = v_1;
    temp2 = beta(1);
    m_p = zeros(n, 1);
    m_pp = zeros(n, 1);
    y = y0;

    for k = 1: (N - 1)
        alpha(k) = v_n' * (Pinv * (A * (Pinv * v_n)));
        temp1 = (Pinv * (A * (Pinv * v_n))) - alpha(k) * v_n - beta(k) * v_p;
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
        end

        gamma(k) = sqrt(gamma_h ^ 2 + beta(k + 1) ^ 2);
        c(k) = gamma_h / gamma(k);
        s(k) = beta(k + 1) / gamma(k);

        m = 1 / gamma(k) * (v_n - epsilon(k) * m_pp - delta(k) * m_p);

        y = y + m * temp2 * c(k);
        temp2 = temp2 * s(k);

        x = Pinv * y;
        r = norm(A * x - b);
        relres = r / norm(b);
        resvec(k + 1) = r;
        iter = k;
        if (r < tol) || (beta(k + 1) < tol)
            flag = 0;
            resvec = resvec(1: (k + 1));
            break
        end

        m_pp = m_p;
        m_p = m;
        v_p = v_n;
        v_n = v_f;

    end  
end