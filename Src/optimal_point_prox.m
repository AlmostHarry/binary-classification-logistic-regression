
clear all;

% 指定.mat文件的名称和路径
file_name = 'a9a.txt.mat'; % 这是保存的.mat文件的名称
filepath = ['./', file_name]; % 假设.mat文件位于当前工作目录

% 加载.mat文件
load(filepath);

% 检查Xtrain和Ylabel是否成功加载
if exist('data', 'var') && iscell(data) && length(data) >= 2
    Xtrain = data{1};
    Ylabel = data{2};
else
    error('.mat 文件中缺少必要的变量 Xtrain 或 Ylabel。');
end
A = Xtrain;
b = Ylabel;

[m, n] = size(A); % 假设文件中已经包含变量 A 和 b

% Parameters
lambda = 1 / (2 * m);
mu = 10^(-2);

% Initial guess
x0 = zeros(m, 1);

% Maximum iterations
max_iter = 5000;
%tol = 1e-6;

[x_prox, k_prox,function_val_prox] = proximal_gradient_method(A, b, lambda, mu, x0, max_iter);

figure;
plot(1:k_prox, function_val_prox(1:k_prox)); % 绘制函数值与迭代步数的关系
set(gca, 'YScale', 'log'); % 如果需要，设置y轴为对数尺度
title('Function Error vs. Iteration (Prox)');
xlabel('Iteration');
ylabel('Function Error');
grid on;

% 函数定义部分

function l = l_function(A, b, x, lambda, mu)
    l = sum(log(1 + exp(-b .* (A' * x)))) / length(b) + lambda * norm(x)^2 + mu * norm(x, 1);
end

function f = f_function(A, b, x, lambda)
    f = sum(log(1 + exp(-b .* (A' * x)))) / length(b) + lambda * norm(x)^2;
end

function x = proximal_operator(v, lambda)
    x = sign(v) .* max(abs(v) - lambda, 0);
end

function grad = compute_gradient(A, b, x, lambda)
    m = length(b);
    grad = zeros(size(A, 1), 1);  % 初始化 grad 为零向量
    for i = 1:m
        grad = grad - A(:,i) * b(i) * (1 - 1 / (1 + exp( - b(i) * A(:, i)' * x))) / m;
    end
    grad = grad + 2 * lambda * x;
end

function x_next = prox_backtracking_line_search(f, x, grad, beta, mu)
    t = 2;
    x_next = proximal_operator(x - t * grad, mu * t);
    while f(x_next) > f(x) + grad' * (x_next - x) + 1/2 / t * norm(x_next - x) ^ 2
        t = beta * t;
        x_next = proximal_operator(x - t * grad, mu * t);
    end
end

function [x, k,function_val_prox] = proximal_gradient_method(A, b, lambda, mu, x0, max_iter)
    x = x0;
    beta = 0.5;
    function_val_prox = zeros(max_iter, 1);
    
    for k = 1:max_iter
        disp(k);
        grad = compute_gradient(A, b, x, lambda);
        x_next = prox_backtracking_line_search(@(x) f_function(A, b, x, lambda), x, grad, beta, mu);
         %format long
        z = l_function(A, b, x_next, lambda, mu);
        disp(z);
       
        %disp(x_next)
        x = x_next;
        function_val_prox(k) = z;
    end
    
    save('x_optimal.mat', 'x');
end