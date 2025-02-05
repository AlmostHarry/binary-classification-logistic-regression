
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
disp(m);
% Parameters
lambda = 1 / (2 * m);
mu = 0.1;

% Initial guess
x0 = zeros(m, 1);

% Maximum iterations
max_iter = 10000;
tol = 1e-6;

%使用近似点梯度法求解问题
[x_prox, k_prox, condition_numbers_prox] = proximal_gradient_method(A, b, lambda, mu, x0, max_iter, tol);

%fisit算法求解问题
[x_fista, k_fista, condition_numbers_fisit] = fista(A, b, lambda, mu, x0, max_iter, tol);

% l函数即是需要求极小值的函数
function l = l_function(A, b, x, lambda, mu)
    l = sum(log(1 + exp(-b .* (A' * x)))) / length(b) + lambda * norm(x)^2 + mu * norm(x, 1);
end

% l函数不计入x的一阶范数即得到f
function f = f_function(A, b, x, lambda)
    f = sum(log(1 + exp(-b .* (A' * x)))) / length(b) + lambda * norm(x)^2;
end

%计算f关于x的梯度，grad即是梯度
function grad = compute_gradient(A, b, x, lambda)
    m = length(b);
    grad = zeros(size(A, 1), 1);  % 初始化 grad 为零向量
    for i = 1:m
        grad = grad - A(:,i) * b(i) * (1 - 1 / (1 + exp( - b(i) * A(:, i)' * x))) / m;
    end
    grad = grad + 2 * lambda * x;
end

%计算prox_{th}(x)，返回值x即是近似点映射之后的值
function x = proximal_operator(v, lambda)
    x = sign(v) .* max(abs(v) - lambda, 0);
end

%近似点梯度法的线搜索
function [t, x_next] = prox_backtracking_line_search(f, x, grad, beta, mu)
    t = 2;
    x_next = proximal_operator(x - t * grad, mu * t);
    while f(x_next) > f(x) + grad' * (x_next - x) + 1/2 / t * norm(x_next - x) ^ 2
        t = beta * t;
        x_next = proximal_operator(x - t * grad, mu * t);
    end
end

%fisit法的线搜索
function [t, x_next] = fisit_backtracking_line_search(f, y, grad, beta, mu)
    t = 2;
    x_next = proximal_operator(y - t * grad, mu * t);
    while f(x_next) > f(y) + grad' * (x_next - y) + 1/2 / t * norm(x_next - y) ^ 2
        t = beta * t;
        x_next = proximal_operator(y - t * grad, mu * t);
    end
end

%近似点梯度法求解问题，k是迭代次数，conditional_numbers_prox是保存条件数的数组。
%每次迭代时判断条件数是否小于tol(1e-6),同时同步更新conditional_numbers_prox中的相应元素
function [x, k, condition_numbers_prox] = proximal_gradient_method(A, b, lambda, mu, x0, max_iter, tol)
    x = x0;%初始点
    beta = 0.8;%线搜索参数
    condition_numbers_prox = zeros(max_iter, 1);
    
    for k = 1:max_iter
        %disp(k);
        grad = compute_gradient(A, b, x, lambda);
        [t, x_next] = prox_backtracking_line_search(@(x) f_function(A, b, x, lambda), x, grad, beta, mu);
        condition_number = norm(x_next - x) / t;
        %disp(condition_number);
        condition_numbers_prox(k) = condition_number; % 条件数
        if condition_number < tol
            break;
        end
        x = x_next;
    end
    %打印稀疏度、迭代步数相关信息
    sparsity = sum(x == 0) / length(x);
    disp('近似点梯度法解的稀疏度：');
    disp(sparsity);
    disp('迭代次数');
    disp(k);
end

%fisit法求解问题，k是迭代次数，conditional_numbers_fisit是保存条件数的数组。
%每次迭代时判断条件数是否小于tol(1e-6),同时同步更新conditional_numbers_fisit中的相应元素
function [x, k, condition_numbers_fisit] = fista(A, b, lambda, mu, x0, max_iter, tol)
    x = x0;%初始点
    y = x0;%初始点
    beta = 0.8;%线搜索参数
    condition_numbers_fisit = zeros(max_iter, 1);
   
    
    for k = 1:max_iter
        %disp(k);
        grad = compute_gradient(A, b, y, lambda);
        [t, x_next] = fisit_backtracking_line_search(@(x) f_function(A, b, x, lambda), y, grad, beta, mu);
        condition_number = norm(x_next - x) / t;
        condition_numbers_fisit(k) = condition_number; % 条件数
        %disp(condition_number);
        if condition_number < tol
            break;
        end
        y = x_next + (k-2)/(k+1)*(x_next - x); 
        x = x_next;
    end
    %打印稀疏度、迭代步数相关信息
    sparsity = sum(x == 0) / length(x);
    disp('fisit法解的稀疏度：');
    disp(sparsity);
    disp('迭代次数');
    disp(k);
end