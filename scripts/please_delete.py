import numpy as np
res_x, res_y = 448, 832

count = 0.0
for i in range(350):
    count = count + 0.01
    a = res_x/count
    b = res_y/count
    print(a,b)
    if( (a/np.floor(a)) == 1.0 and b/np.floor(b) == 1.0 ):
        print(count)
        print(a,b)


