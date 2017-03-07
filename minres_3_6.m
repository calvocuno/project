clear all;
clc;
%%
A = [1, -0.65, -0.55;
    -0.25, 0.95, -0.10;
    -0.15, -0.05, 1];
A = A + A';
b = [5; 10; 0];
N = 10;
%%
n = length(b);
beta = zeros(N, 1);
alpha = zeros(N, 1);
gamma = zeros(N, 1);
delta = zeros(N, 1);
epsilon = zeros(N, 1);
c = zeros(N, 1);
s = zeros(N, 1);
%%
v_0 = zeros(n, 1);
beta(1) = norm(b);
v_1 = b / beta(1);
v(:, 1) = v_1; 
%%
v_p = v_0;
v_n = v_1;
gamma_h = v_1' * A * v_1;
temp2 = beta(1);
m_p = zeros(n, 1);
m_pp = zeros(n, 1);
x = 0;

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

    x = x + m * temp2 * c(k)
    temp2 = temp2 * s(k);
    
    norm(A * x - b)
    
    temp2
    
    m_pp = m_p;
    m_p = m;
    v_p = v_n;
    v_n = v_f;
end  
%%    