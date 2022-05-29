clear all;
% x(t+1) = Ax(t) + w
% y(t) = d(x(t)) + v
% w ~ N(0, Q)
% v ~ N(0, R)

%% Parameters
A = 1.001; % transition matrix
Q = 0.25; % covariance of noise w
R = 0.3; % covariance of noise v
x0 = 0; % initial state
len = 200; % length of the sequence
N = 5000; % number of particle

%% Initialization
x_true = zeros(1, len);
x_pred = zeros(1, len);
y = zeros(1, len);
x_true(1) = x0;
x_pred(1) = x0;
seed = 7;

%% Preparation of true location
randn('state', seed);
y(1) = d(x_true(1)) + normrnd(0, R^0.5);

for i = 2:len
    x_true(i) = A * x_true(i - 1) + normrnd(0, Q^0.5);
    y(i) = d(x_true(i)) + normrnd(0, R^0.5);
end

%% Prediction
particle = zeros(1, N) + x0; % initialize particles with x0
w = zeros(1, N) + 1 / N; % initialize weights with 1/N

for i = 2:len
    disp(i)
    % p(x(t)|x(t-1))
    particle = A .* particle + normrnd(0, Q^0.5, [1, N]);
    % w = p(y|x)*(1/N)
    w = exp(- (y(i) - d(particle)).^2 / (2 * R)) / sqrt(2 * pi * R);
    % normalization using 1/N
    w = w ./ sum(w);
    % resampling and update particle
    for n = 1:N
        particle(n) = particle(find(rand <= cumsum(w), 1));
    end

    % estimated variable
    x_pred(i) = mean(particle);
end

%% Evaluation
mse = sum((x_pred - x_true).^2) ./ size(x_true, 2);
disp(['MSE:', num2str(mse)]);

%% Illustration
plot(x_pred, '-', 'linewidth', 1.5)
hold on
plot(x_true, '--', 'linewidth', 1.5)
xlabel('time step');
ylabel('x');
legend('prediction of x', 'true x');
grid on;
title(['Particle Filter (x0=', num2str(x0), ' A=', num2str(A),' N=', num2str(N), ')']);
saveas(gcf, 'Particle.png');

function depth = d(x)
    depth = sin(x) + x ./ (1 + x.^2);
%     depth = 108 * x;
end
