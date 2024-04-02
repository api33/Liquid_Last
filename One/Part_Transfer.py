def funcTransfer(T_nums,Lines_Points,boundary):
    #print("get",Lines_Points)
    length1=len(Lines_Points)
    length2=len(Lines_Points[0])
    a,b,c,d=boundary
    assert a<=b<=c<=d
    a0,b0,a1,b1=T_nums
    a0,b0,a1,b1,a,b = map(float, [a0,b0,a1,b1,a,b])
    #print(T_nums)
    #print("长度为",length1,len(Lines_Points[0]),len(Lines_Points[1]))

    for i in range(length1):
        for j in range(length2):
            currgray=Lines_Points[i][j]
            if currgray <=a:
                currgray=88

            elif currgray>=d:
                currgray=168
            elif a<currgray<=b:
                currgray=a0*(currgray-a)+b0
            elif b<currgray<=c:
                currgray=128
            elif c<currgray<d:
                currgray=a1*(currgray-c)+b1
            Lines_Points[i][j]=round(currgray, 2)

    return Lines_Points

