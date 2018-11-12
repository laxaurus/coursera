X= [1 0 0; 1 0 1; 1 1 0;1 1 1];
t1 = [-30 20 20 ; 10 -20 -20];
t2 = [-10 20 20]

a2 = [ones(rows(X), 1) sigmoid(X * t1')]
a3 = round(sigmoid(a2 * t2'))
 