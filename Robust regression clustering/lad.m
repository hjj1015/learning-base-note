function x = lad(A, b, rho, alpha)
% lad  Least absolute deviations fitting via ADMM
%
% [x, history] = lad(A, b, rho, alpha)
%
% Solves the following problem via ADMM:
%
%   minimize     ||Ax - b||_1
%
% The solution is returned in the vector x.
% A is the data
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;
%% Global constants and defaults
QUIET    = 0;
MAX_ITER = 100;   %之前是1000次
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(A);
%% ADMM solver
x = zeros(n,1);   %初始解x
z = zeros(m,1);   
u = zeros(m,1);   

% if ~QUIET
%     fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
%       'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
% end

for k = 1:MAX_ITER

    if k > 1
        x = R \ (R' \ (A'*(b + z - u)));
    else
        try
            R = chol(A'*A);
            x = R \ (R' \ (A'*(b + z - u)));
        catch
%             x = randn(n,1);  % A \in R^{m,n}
            r = det(A'*A);
            fprintf('线性方程系数矩阵的行列式为%d\n',r);
            error('ADMM求解LAD时，线性方程系数矩阵奇异!');
        end     
    end

    zold = z;
    Ax_hat = alpha*A*x + (1-alpha)*(zold + b);
    z = shrinkage(Ax_hat - b + u, 1/rho);  % x初始值是A'Ax=A'b的解，这时z=u=0

    u = u + (Ax_hat - z - b);  % 这个u是增广拉格朗日系数，它也需要更新

    % diagnostics, reporting, termination checks

%     objval  = objective(z);

    r_norm  = norm(A*x - z - b);
    s_norm  = norm(-rho*A'*(z - zold));

    eps_pri = sqrt(m)*ABSTOL + RELTOL*max([norm(A*x), norm(-z), norm(b)]);
    eps_dual = sqrt(n)*ABSTOL + RELTOL*norm(rho*A'*u);

%     if ~QUIET
%         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), history.objval(k));
%     end

    if (r_norm < eps_pri && ...
       s_norm < eps_dual)
         break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(z)
    %返回z的L_1范数
    obj = norm(z,1);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end