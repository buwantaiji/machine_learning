%x = [0:0.02:2*pi];
%y1 = sin(x);
%y2 = cos(x);
%printf("heheda!\n");
%hold on;
%plot(x, y1, "r", "linewidth", 2)
%plot(x, y2, "b", "linewidth", 2)
%hold off;

data = load("ex2data1.txt");
X = data(:, 1:2);
y = data(:, 3);
indices1 = find(y == 1);
indices2 = find(y == 0);
hold on;
plot(X(indices1, 1), X(indices1, 2), "r*", "markersize", 7)
plot(X(indices2, 1), X(indices2, 2), "bo", "markersize", 7)
