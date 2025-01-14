#@title mybets { form-width: "100px" }
#ipopt code for generating the fourth order 16 stage
#from pyomo.core.expr import current as EXPR
import pyomo.core.expr as EXPR
import pyomo.environ as pyo
from pyomo.environ import *

def mybets(m):

    c=[[],[]]
    Q=[]
    zval=-m.beta
    for i in range(m.s.value): # initialise internal stability functions
        Q.append(m.b[i]*zval)

    for J in range(1,m.s.value): # determine A^{J=1..s-1}=>beta_{J+1} and store in inew
    #for J in range(1,int(m.order)): # determine A^{J=1..s-1}=>beta_{J+1} and store in inew
    #for J in range(0,1): # determine J+1 and store in inew
        ibet=J+1
        iold=(J-1)%2
        inew=J%2
        #print(iold,inew)
        for i in range(J):  #dead rows
            c[inew].append([None])

        #Populate c[inew]
        for i in range(J,m.s.value):  #live rows in A^J
            r1=[]
            for j in range(i-J+1): # live cols in A^J
                if J>1:
                    c1=0
                    for k in range(j+1,i-J+2): # non-zero vals: c[iold] in row i has non zero up to c[iold][i][i-J+1], a in col j has non zero from a[j+1,j]
                        #print(i,j,k)
                        c1+=c[iold][i][k]*m.a[k,j]
                else:
                    c1=m.a[i,j]   # initialise to A
                r1.append(c1)
            c[inew].append(r1)

        #Reset c[iold]
        c[iold]=[]

        #Use c[inew] to determine beta expression: first sum over row and then apply beta
        #csos23  ARE THESE MAD??
        if ibet>m.order:
            y1=0 # bet constraint expression
            for i in range(J,m.s.value): # live rows in A^J:   J to s-1 live rows (s-J live)
                rowsum=0
                for j in range(i-J+1):  # live cols in row i  of A^J (s-J-j live entries)
                    rowsum+=c[inew][i][j]
                y1+=m.b[i]*rowsum
            y1-=m.bet[ibet]

            m.cons.add(y1 == 0 )

            #print('**beta**',ibet,m.bet[ibet],EXPR.expression_to_string(y1),'==0\n')
            print('**beta**',ibet,m.bet[ibet])

        #Use c[inew] to determine internal stability functions evaluated at zval
        zpow=zval**(J+1)
        for j in range(m.s.value-J): # live cols in A^J:   0 to s-J-1 live cols (s-J live)
            colsum=0
            for i in range(J+j,m.s.value):  # live rows in col j  of A^J (s-J-j live entries)
                colsum+=m.b[i]*c[inew][i][j]
            Q[j]+=colsum*zpow #accumulate stability polynomials


    #csos23
    norm=0.0
    L=m.s.value
    #L=2
    for i in range(m.s.value): # initialise internal stability objective function
    #g=6
    #for i in range(g,g+1): # initialise internal stability objective function
        q1=abs(Q[i])
        #q1=Q[i]
        norm+=q1**L
    ##csos23
    ob=norm**(1/L)
    #ob=norm
    #print('Q',i,EXPR.expression_to_string(Q[i]),'\n')


    ##ALTERNATE OBJECTIVE FUNCTION
    norm=0.0
    L=m.s.value
    #L=2
    for i in range(1,m.s.value):
        for j in range(i):
            a1=abs(m.a[i,j])
            norm+=a1**L
    for j in m.cols:
        b1=abs(m.b[j])
        norm+=b1**L
    #csos
    ob=norm**(1/L)


    #ASSIGNING ob to objective function
    m.obj = pyo.Objective(expr=ob)

#@title mycons { form-width: "100px" }
import pyomo.environ as pyo
def mycons(m):

    c=[0]
    c.append(sum(m.a[1,i] for i in range(1)))
    c.append(sum(m.a[2,i] for i in range(2)))
    c.append(sum(m.a[3,i] for i in range(3)))
    c.append(sum(m.a[4,i] for i in range(4)))
    c.append(sum(m.a[5,i] for i in range(5)))
    c.append(sum(m.a[6,i] for i in range(6)))
    c.append(sum(m.a[7,i] for i in range(7)))
    c.append(sum(m.a[8,i] for i in range(8)))
    c.append(sum(m.a[9,i] for i in range(9)))
    c.append(sum(m.a[10,i] for i in range(10)))
    c.append(sum(m.a[11,i] for i in range(11)))
    c.append(sum(m.a[12,i] for i in range(12)))
    c.append(sum(m.a[13,i] for i in range(13)))
    c.append(sum(m.a[14,i] for i in range(14)))
    c.append(sum(m.a[15,i] for i in range(15)))


    y1 = +m.b[0]+m.b[1]+m.b[2]+m.b[3]+m.b[4]+m.b[5]+m.b[6]+m.b[7]+m.b[8]+m.b[9]+m.b[10]+m.b[11]+m.b[12]+m.b[13]+m.b[14]+m.b[15] - (1/1)
    m.cons.add(y1 == 0 )

    y1 = +m.b[1]*c[1]+m.b[2]*c[2]+m.b[3]*c[3]+m.b[4]*c[4]+m.b[5]*c[5]+m.b[6]*c[6]+m.b[7]*c[7]+m.b[8]*c[8]+m.b[9]*c[9]+m.b[10]*c[10]+m.b[11]*c[11]+m.b[12]*c[12]+m.b[13]*c[13]+m.b[14]*c[14]+m.b[15]*c[15] - (1/2)
    m.cons.add(y1 == 0 )

    y1 = +m.b[1]*c[1]**2+m.b[2]*c[2]**2+m.b[3]*c[3]**2+m.b[4]*c[4]**2+m.b[5]*c[5]**2+m.b[6]*c[6]**2+m.b[7]*c[7]**2+m.b[8]*c[8]**2+m.b[9]*c[9]**2+m.b[10]*c[10]**2+m.b[11]*c[11]**2+m.b[12]*c[12]**2+m.b[13]*c[13]**2+m.b[14]*c[14]**2+m.b[15]*c[15]**2 - (1/3)
    m.cons.add(y1 == 0 )

    y1 = +m.b[2]*m.a[2,1]*c[1]+m.b[3]*m.a[3,1]*c[1]+m.b[3]*m.a[3,2]*c[2]+m.b[4]*m.a[4,1]*c[1]+m.b[4]*m.a[4,2]*c[2]+m.b[4]*m.a[4,3]*c[3]+m.b[5]*m.a[5,1]*c[1]+m.b[5]*m.a[5,2]*c[2]+m.b[5]*m.a[5,3]*c[3]+m.b[5]*m.a[5,4]*c[4]+m.b[6]*m.a[6,1]*c[1]+m.b[6]*m.a[6,2]*c[2]+m.b[6]*m.a[6,3]*c[3]+m.b[6]*m.a[6,4]*c[4]+m.b[6]*m.a[6,5]*c[5]+m.b[7]*m.a[7,1]*c[1]+m.b[7]*m.a[7,2]*c[2]+m.b[7]*m.a[7,3]*c[3]+m.b[7]*m.a[7,4]*c[4]+m.b[7]*m.a[7,5]*c[5]+m.b[7]*m.a[7,6]*c[6]+m.b[8]*m.a[8,1]*c[1]+m.b[8]*m.a[8,2]*c[2]+m.b[8]*m.a[8,3]*c[3]+m.b[8]*m.a[8,4]*c[4]+m.b[8]*m.a[8,5]*c[5]+m.b[8]*m.a[8,6]*c[6]+m.b[8]*m.a[8,7]*c[7]+m.b[9]*m.a[9,1]*c[1]+m.b[9]*m.a[9,2]*c[2]+m.b[9]*m.a[9,3]*c[3]+m.b[9]*m.a[9,4]*c[4]+m.b[9]*m.a[9,5]*c[5]+m.b[9]*m.a[9,6]*c[6]+m.b[9]*m.a[9,7]*c[7]+m.b[9]*m.a[9,8]*c[8]+m.b[10]*m.a[10,1]*c[1]+m.b[10]*m.a[10,2]*c[2]+m.b[10]*m.a[10,3]*c[3]+m.b[10]*m.a[10,4]*c[4]+m.b[10]*m.a[10,5]*c[5]+m.b[10]*m.a[10,6]*c[6]+m.b[10]*m.a[10,7]*c[7]+m.b[10]*m.a[10,8]*c[8]+m.b[10]*m.a[10,9]*c[9]+m.b[11]*m.a[11,1]*c[1]+m.b[11]*m.a[11,2]*c[2]+m.b[11]*m.a[11,3]*c[3]+m.b[11]*m.a[11,4]*c[4]+m.b[11]*m.a[11,5]*c[5]+m.b[11]*m.a[11,6]*c[6]+m.b[11]*m.a[11,7]*c[7]+m.b[11]*m.a[11,8]*c[8]+m.b[11]*m.a[11,9]*c[9]+m.b[11]*m.a[11,10]*c[10]+m.b[12]*m.a[12,1]*c[1]+m.b[12]*m.a[12,2]*c[2]+m.b[12]*m.a[12,3]*c[3]+m.b[12]*m.a[12,4]*c[4]+m.b[12]*m.a[12,5]*c[5]+m.b[12]*m.a[12,6]*c[6]+m.b[12]*m.a[12,7]*c[7]+m.b[12]*m.a[12,8]*c[8]+m.b[12]*m.a[12,9]*c[9]+m.b[12]*m.a[12,10]*c[10]+m.b[12]*m.a[12,11]*c[11]+m.b[13]*m.a[13,1]*c[1]+m.b[13]*m.a[13,2]*c[2]+m.b[13]*m.a[13,3]*c[3]+m.b[13]*m.a[13,4]*c[4]+m.b[13]*m.a[13,5]*c[5]+m.b[13]*m.a[13,6]*c[6]+m.b[13]*m.a[13,7]*c[7]+m.b[13]*m.a[13,8]*c[8]+m.b[13]*m.a[13,9]*c[9]+m.b[13]*m.a[13,10]*c[10]+m.b[13]*m.a[13,11]*c[11]+m.b[13]*m.a[13,12]*c[12]+m.b[14]*m.a[14,1]*c[1]+m.b[14]*m.a[14,2]*c[2]+m.b[14]*m.a[14,3]*c[3]+m.b[14]*m.a[14,4]*c[4]+m.b[14]*m.a[14,5]*c[5]+m.b[14]*m.a[14,6]*c[6]+m.b[14]*m.a[14,7]*c[7]+m.b[14]*m.a[14,8]*c[8]+m.b[14]*m.a[14,9]*c[9]+m.b[14]*m.a[14,10]*c[10]+m.b[14]*m.a[14,11]*c[11]+m.b[14]*m.a[14,12]*c[12]+m.b[14]*m.a[14,13]*c[13]+m.b[15]*m.a[15,1]*c[1]+m.b[15]*m.a[15,2]*c[2]+m.b[15]*m.a[15,3]*c[3]+m.b[15]*m.a[15,4]*c[4]+m.b[15]*m.a[15,5]*c[5]+m.b[15]*m.a[15,6]*c[6]+m.b[15]*m.a[15,7]*c[7]+m.b[15]*m.a[15,8]*c[8]+m.b[15]*m.a[15,9]*c[9]+m.b[15]*m.a[15,10]*c[10]+m.b[15]*m.a[15,11]*c[11]+m.b[15]*m.a[15,12]*c[12]+m.b[15]*m.a[15,13]*c[13]+m.b[15]*m.a[15,14]*c[14] - (1/6)
    m.cons.add(y1 == 0 )

    y1 = +m.b[1]*c[1]**3+m.b[2]*c[2]**3+m.b[3]*c[3]**3+m.b[4]*c[4]**3+m.b[5]*c[5]**3+m.b[6]*c[6]**3+m.b[7]*c[7]**3+m.b[8]*c[8]**3+m.b[9]*c[9]**3+m.b[10]*c[10]**3+m.b[11]*c[11]**3+m.b[12]*c[12]**3+m.b[13]*c[13]**3+m.b[14]*c[14]**3+m.b[15]*c[15]**3 - (1/4)
    m.cons.add(y1 == 0 )

    y1 = +m.b[2]*m.a[2,1]*c[2]*c[1]+m.b[3]*m.a[3,1]*c[3]*c[1]+m.b[3]*m.a[3,2]*c[3]*c[2]+m.b[4]*m.a[4,1]*c[4]*c[1]+m.b[4]*m.a[4,2]*c[4]*c[2]+m.b[4]*m.a[4,3]*c[4]*c[3]+m.b[5]*m.a[5,1]*c[5]*c[1]+m.b[5]*m.a[5,2]*c[5]*c[2]+m.b[5]*m.a[5,3]*c[5]*c[3]+m.b[5]*m.a[5,4]*c[5]*c[4]+m.b[6]*m.a[6,1]*c[6]*c[1]+m.b[6]*m.a[6,2]*c[6]*c[2]+m.b[6]*m.a[6,3]*c[6]*c[3]+m.b[6]*m.a[6,4]*c[6]*c[4]+m.b[6]*m.a[6,5]*c[6]*c[5]+m.b[7]*m.a[7,1]*c[7]*c[1]+m.b[7]*m.a[7,2]*c[7]*c[2]+m.b[7]*m.a[7,3]*c[7]*c[3]+m.b[7]*m.a[7,4]*c[7]*c[4]+m.b[7]*m.a[7,5]*c[7]*c[5]+m.b[7]*m.a[7,6]*c[7]*c[6]+m.b[8]*m.a[8,1]*c[8]*c[1]+m.b[8]*m.a[8,2]*c[8]*c[2]+m.b[8]*m.a[8,3]*c[8]*c[3]+m.b[8]*m.a[8,4]*c[8]*c[4]+m.b[8]*m.a[8,5]*c[8]*c[5]+m.b[8]*m.a[8,6]*c[8]*c[6]+m.b[8]*m.a[8,7]*c[8]*c[7]+m.b[9]*m.a[9,1]*c[9]*c[1]+m.b[9]*m.a[9,2]*c[9]*c[2]+m.b[9]*m.a[9,3]*c[9]*c[3]+m.b[9]*m.a[9,4]*c[9]*c[4]+m.b[9]*m.a[9,5]*c[9]*c[5]+m.b[9]*m.a[9,6]*c[9]*c[6]+m.b[9]*m.a[9,7]*c[9]*c[7]+m.b[9]*m.a[9,8]*c[9]*c[8]+m.b[10]*m.a[10,1]*c[10]*c[1]+m.b[10]*m.a[10,2]*c[10]*c[2]+m.b[10]*m.a[10,3]*c[10]*c[3]+m.b[10]*m.a[10,4]*c[10]*c[4]+m.b[10]*m.a[10,5]*c[10]*c[5]+m.b[10]*m.a[10,6]*c[10]*c[6]+m.b[10]*m.a[10,7]*c[10]*c[7]+m.b[10]*m.a[10,8]*c[10]*c[8]+m.b[10]*m.a[10,9]*c[10]*c[9]+m.b[11]*m.a[11,1]*c[11]*c[1]+m.b[11]*m.a[11,2]*c[11]*c[2]+m.b[11]*m.a[11,3]*c[11]*c[3]+m.b[11]*m.a[11,4]*c[11]*c[4]+m.b[11]*m.a[11,5]*c[11]*c[5]+m.b[11]*m.a[11,6]*c[11]*c[6]+m.b[11]*m.a[11,7]*c[11]*c[7]+m.b[11]*m.a[11,8]*c[11]*c[8]+m.b[11]*m.a[11,9]*c[11]*c[9]+m.b[11]*m.a[11,10]*c[11]*c[10]+m.b[12]*m.a[12,1]*c[12]*c[1]+m.b[12]*m.a[12,2]*c[12]*c[2]+m.b[12]*m.a[12,3]*c[12]*c[3]+m.b[12]*m.a[12,4]*c[12]*c[4]+m.b[12]*m.a[12,5]*c[12]*c[5]+m.b[12]*m.a[12,6]*c[12]*c[6]+m.b[12]*m.a[12,7]*c[12]*c[7]+m.b[12]*m.a[12,8]*c[12]*c[8]+m.b[12]*m.a[12,9]*c[12]*c[9]+m.b[12]*m.a[12,10]*c[12]*c[10]+m.b[12]*m.a[12,11]*c[12]*c[11]+m.b[13]*m.a[13,1]*c[13]*c[1]+m.b[13]*m.a[13,2]*c[13]*c[2]+m.b[13]*m.a[13,3]*c[13]*c[3]+m.b[13]*m.a[13,4]*c[13]*c[4]+m.b[13]*m.a[13,5]*c[13]*c[5]+m.b[13]*m.a[13,6]*c[13]*c[6]+m.b[13]*m.a[13,7]*c[13]*c[7]+m.b[13]*m.a[13,8]*c[13]*c[8]+m.b[13]*m.a[13,9]*c[13]*c[9]+m.b[13]*m.a[13,10]*c[13]*c[10]+m.b[13]*m.a[13,11]*c[13]*c[11]+m.b[13]*m.a[13,12]*c[13]*c[12]+m.b[14]*m.a[14,1]*c[14]*c[1]+m.b[14]*m.a[14,2]*c[14]*c[2]+m.b[14]*m.a[14,3]*c[14]*c[3]+m.b[14]*m.a[14,4]*c[14]*c[4]+m.b[14]*m.a[14,5]*c[14]*c[5]+m.b[14]*m.a[14,6]*c[14]*c[6]+m.b[14]*m.a[14,7]*c[14]*c[7]+m.b[14]*m.a[14,8]*c[14]*c[8]+m.b[14]*m.a[14,9]*c[14]*c[9]+m.b[14]*m.a[14,10]*c[14]*c[10]+m.b[14]*m.a[14,11]*c[14]*c[11]+m.b[14]*m.a[14,12]*c[14]*c[12]+m.b[14]*m.a[14,13]*c[14]*c[13]+m.b[15]*m.a[15,1]*c[15]*c[1]+m.b[15]*m.a[15,2]*c[15]*c[2]+m.b[15]*m.a[15,3]*c[15]*c[3]+m.b[15]*m.a[15,4]*c[15]*c[4]+m.b[15]*m.a[15,5]*c[15]*c[5]+m.b[15]*m.a[15,6]*c[15]*c[6]+m.b[15]*m.a[15,7]*c[15]*c[7]+m.b[15]*m.a[15,8]*c[15]*c[8]+m.b[15]*m.a[15,9]*c[15]*c[9]+m.b[15]*m.a[15,10]*c[15]*c[10]+m.b[15]*m.a[15,11]*c[15]*c[11]+m.b[15]*m.a[15,12]*c[15]*c[12]+m.b[15]*m.a[15,13]*c[15]*c[13]+m.b[15]*m.a[15,14]*c[15]*c[14] - (1/8)
    m.cons.add(y1 == 0 )

    y1 = +m.b[2]*m.a[2,1]*c[1]**2+m.b[3]*m.a[3,1]*c[1]**2+m.b[3]*m.a[3,2]*c[2]**2+m.b[4]*m.a[4,1]*c[1]**2+m.b[4]*m.a[4,2]*c[2]**2+m.b[4]*m.a[4,3]*c[3]**2+m.b[5]*m.a[5,1]*c[1]**2+m.b[5]*m.a[5,2]*c[2]**2+m.b[5]*m.a[5,3]*c[3]**2+m.b[5]*m.a[5,4]*c[4]**2+m.b[6]*m.a[6,1]*c[1]**2+m.b[6]*m.a[6,2]*c[2]**2+m.b[6]*m.a[6,3]*c[3]**2+m.b[6]*m.a[6,4]*c[4]**2+m.b[6]*m.a[6,5]*c[5]**2+m.b[7]*m.a[7,1]*c[1]**2+m.b[7]*m.a[7,2]*c[2]**2+m.b[7]*m.a[7,3]*c[3]**2+m.b[7]*m.a[7,4]*c[4]**2+m.b[7]*m.a[7,5]*c[5]**2+m.b[7]*m.a[7,6]*c[6]**2+m.b[8]*m.a[8,1]*c[1]**2+m.b[8]*m.a[8,2]*c[2]**2+m.b[8]*m.a[8,3]*c[3]**2+m.b[8]*m.a[8,4]*c[4]**2+m.b[8]*m.a[8,5]*c[5]**2+m.b[8]*m.a[8,6]*c[6]**2+m.b[8]*m.a[8,7]*c[7]**2+m.b[9]*m.a[9,1]*c[1]**2+m.b[9]*m.a[9,2]*c[2]**2+m.b[9]*m.a[9,3]*c[3]**2+m.b[9]*m.a[9,4]*c[4]**2+m.b[9]*m.a[9,5]*c[5]**2+m.b[9]*m.a[9,6]*c[6]**2+m.b[9]*m.a[9,7]*c[7]**2+m.b[9]*m.a[9,8]*c[8]**2+m.b[10]*m.a[10,1]*c[1]**2+m.b[10]*m.a[10,2]*c[2]**2+m.b[10]*m.a[10,3]*c[3]**2+m.b[10]*m.a[10,4]*c[4]**2+m.b[10]*m.a[10,5]*c[5]**2+m.b[10]*m.a[10,6]*c[6]**2+m.b[10]*m.a[10,7]*c[7]**2+m.b[10]*m.a[10,8]*c[8]**2+m.b[10]*m.a[10,9]*c[9]**2+m.b[11]*m.a[11,1]*c[1]**2+m.b[11]*m.a[11,2]*c[2]**2+m.b[11]*m.a[11,3]*c[3]**2+m.b[11]*m.a[11,4]*c[4]**2+m.b[11]*m.a[11,5]*c[5]**2+m.b[11]*m.a[11,6]*c[6]**2+m.b[11]*m.a[11,7]*c[7]**2+m.b[11]*m.a[11,8]*c[8]**2+m.b[11]*m.a[11,9]*c[9]**2+m.b[11]*m.a[11,10]*c[10]**2+m.b[12]*m.a[12,1]*c[1]**2+m.b[12]*m.a[12,2]*c[2]**2+m.b[12]*m.a[12,3]*c[3]**2+m.b[12]*m.a[12,4]*c[4]**2+m.b[12]*m.a[12,5]*c[5]**2+m.b[12]*m.a[12,6]*c[6]**2+m.b[12]*m.a[12,7]*c[7]**2+m.b[12]*m.a[12,8]*c[8]**2+m.b[12]*m.a[12,9]*c[9]**2+m.b[12]*m.a[12,10]*c[10]**2+m.b[12]*m.a[12,11]*c[11]**2+m.b[13]*m.a[13,1]*c[1]**2+m.b[13]*m.a[13,2]*c[2]**2+m.b[13]*m.a[13,3]*c[3]**2+m.b[13]*m.a[13,4]*c[4]**2+m.b[13]*m.a[13,5]*c[5]**2+m.b[13]*m.a[13,6]*c[6]**2+m.b[13]*m.a[13,7]*c[7]**2+m.b[13]*m.a[13,8]*c[8]**2+m.b[13]*m.a[13,9]*c[9]**2+m.b[13]*m.a[13,10]*c[10]**2+m.b[13]*m.a[13,11]*c[11]**2+m.b[13]*m.a[13,12]*c[12]**2+m.b[14]*m.a[14,1]*c[1]**2+m.b[14]*m.a[14,2]*c[2]**2+m.b[14]*m.a[14,3]*c[3]**2+m.b[14]*m.a[14,4]*c[4]**2+m.b[14]*m.a[14,5]*c[5]**2+m.b[14]*m.a[14,6]*c[6]**2+m.b[14]*m.a[14,7]*c[7]**2+m.b[14]*m.a[14,8]*c[8]**2+m.b[14]*m.a[14,9]*c[9]**2+m.b[14]*m.a[14,10]*c[10]**2+m.b[14]*m.a[14,11]*c[11]**2+m.b[14]*m.a[14,12]*c[12]**2+m.b[14]*m.a[14,13]*c[13]**2+m.b[15]*m.a[15,1]*c[1]**2+m.b[15]*m.a[15,2]*c[2]**2+m.b[15]*m.a[15,3]*c[3]**2+m.b[15]*m.a[15,4]*c[4]**2+m.b[15]*m.a[15,5]*c[5]**2+m.b[15]*m.a[15,6]*c[6]**2+m.b[15]*m.a[15,7]*c[7]**2+m.b[15]*m.a[15,8]*c[8]**2+m.b[15]*m.a[15,9]*c[9]**2+m.b[15]*m.a[15,10]*c[10]**2+m.b[15]*m.a[15,11]*c[11]**2+m.b[15]*m.a[15,12]*c[12]**2+m.b[15]*m.a[15,13]*c[13]**2+m.b[15]*m.a[15,14]*c[14]**2 - (1/12)
    m.cons.add(y1 == 0 )

    y1 = +m.b[3]*m.a[3,2]*m.a[2,1]*c[1]+m.b[4]*m.a[4,2]*m.a[2,1]*c[1]+m.b[4]*m.a[4,3]*m.a[3,1]*c[1]+m.b[4]*m.a[4,3]*m.a[3,2]*c[2]+m.b[5]*m.a[5,2]*m.a[2,1]*c[1]+m.b[5]*m.a[5,3]*m.a[3,1]*c[1]+m.b[5]*m.a[5,3]*m.a[3,2]*c[2]+m.b[5]*m.a[5,4]*m.a[4,1]*c[1]+m.b[5]*m.a[5,4]*m.a[4,2]*c[2]+m.b[5]*m.a[5,4]*m.a[4,3]*c[3]+m.b[6]*m.a[6,2]*m.a[2,1]*c[1]+m.b[6]*m.a[6,3]*m.a[3,1]*c[1]+m.b[6]*m.a[6,3]*m.a[3,2]*c[2]+m.b[6]*m.a[6,4]*m.a[4,1]*c[1]+m.b[6]*m.a[6,4]*m.a[4,2]*c[2]+m.b[6]*m.a[6,4]*m.a[4,3]*c[3]+m.b[6]*m.a[6,5]*m.a[5,1]*c[1]+m.b[6]*m.a[6,5]*m.a[5,2]*c[2]+m.b[6]*m.a[6,5]*m.a[5,3]*c[3]+m.b[6]*m.a[6,5]*m.a[5,4]*c[4]+m.b[7]*m.a[7,2]*m.a[2,1]*c[1]+m.b[7]*m.a[7,3]*m.a[3,1]*c[1]+m.b[7]*m.a[7,3]*m.a[3,2]*c[2]+m.b[7]*m.a[7,4]*m.a[4,1]*c[1]+m.b[7]*m.a[7,4]*m.a[4,2]*c[2]+m.b[7]*m.a[7,4]*m.a[4,3]*c[3]+m.b[7]*m.a[7,5]*m.a[5,1]*c[1]+m.b[7]*m.a[7,5]*m.a[5,2]*c[2]+m.b[7]*m.a[7,5]*m.a[5,3]*c[3]+m.b[7]*m.a[7,5]*m.a[5,4]*c[4]+m.b[7]*m.a[7,6]*m.a[6,1]*c[1]+m.b[7]*m.a[7,6]*m.a[6,2]*c[2]+m.b[7]*m.a[7,6]*m.a[6,3]*c[3]+m.b[7]*m.a[7,6]*m.a[6,4]*c[4]+m.b[7]*m.a[7,6]*m.a[6,5]*c[5]+m.b[8]*m.a[8,2]*m.a[2,1]*c[1]+m.b[8]*m.a[8,3]*m.a[3,1]*c[1]+m.b[8]*m.a[8,3]*m.a[3,2]*c[2]+m.b[8]*m.a[8,4]*m.a[4,1]*c[1]+m.b[8]*m.a[8,4]*m.a[4,2]*c[2]+m.b[8]*m.a[8,4]*m.a[4,3]*c[3]+m.b[8]*m.a[8,5]*m.a[5,1]*c[1]+m.b[8]*m.a[8,5]*m.a[5,2]*c[2]+m.b[8]*m.a[8,5]*m.a[5,3]*c[3]+m.b[8]*m.a[8,5]*m.a[5,4]*c[4]+m.b[8]*m.a[8,6]*m.a[6,1]*c[1]+m.b[8]*m.a[8,6]*m.a[6,2]*c[2]+m.b[8]*m.a[8,6]*m.a[6,3]*c[3]+m.b[8]*m.a[8,6]*m.a[6,4]*c[4]+m.b[8]*m.a[8,6]*m.a[6,5]*c[5]+m.b[8]*m.a[8,7]*m.a[7,1]*c[1]+m.b[8]*m.a[8,7]*m.a[7,2]*c[2]+m.b[8]*m.a[8,7]*m.a[7,3]*c[3]+m.b[8]*m.a[8,7]*m.a[7,4]*c[4]+m.b[8]*m.a[8,7]*m.a[7,5]*c[5]+m.b[8]*m.a[8,7]*m.a[7,6]*c[6]+m.b[9]*m.a[9,2]*m.a[2,1]*c[1]+m.b[9]*m.a[9,3]*m.a[3,1]*c[1]+m.b[9]*m.a[9,3]*m.a[3,2]*c[2]+m.b[9]*m.a[9,4]*m.a[4,1]*c[1]+m.b[9]*m.a[9,4]*m.a[4,2]*c[2]+m.b[9]*m.a[9,4]*m.a[4,3]*c[3]+m.b[9]*m.a[9,5]*m.a[5,1]*c[1]+m.b[9]*m.a[9,5]*m.a[5,2]*c[2]+m.b[9]*m.a[9,5]*m.a[5,3]*c[3]+m.b[9]*m.a[9,5]*m.a[5,4]*c[4]+m.b[9]*m.a[9,6]*m.a[6,1]*c[1]+m.b[9]*m.a[9,6]*m.a[6,2]*c[2]+m.b[9]*m.a[9,6]*m.a[6,3]*c[3]+m.b[9]*m.a[9,6]*m.a[6,4]*c[4]+m.b[9]*m.a[9,6]*m.a[6,5]*c[5]+m.b[9]*m.a[9,7]*m.a[7,1]*c[1]+m.b[9]*m.a[9,7]*m.a[7,2]*c[2]+m.b[9]*m.a[9,7]*m.a[7,3]*c[3]+m.b[9]*m.a[9,7]*m.a[7,4]*c[4]+m.b[9]*m.a[9,7]*m.a[7,5]*c[5]+m.b[9]*m.a[9,7]*m.a[7,6]*c[6]+m.b[9]*m.a[9,8]*m.a[8,1]*c[1]+m.b[9]*m.a[9,8]*m.a[8,2]*c[2]+m.b[9]*m.a[9,8]*m.a[8,3]*c[3]+m.b[9]*m.a[9,8]*m.a[8,4]*c[4]+m.b[9]*m.a[9,8]*m.a[8,5]*c[5]+m.b[9]*m.a[9,8]*m.a[8,6]*c[6]+m.b[9]*m.a[9,8]*m.a[8,7]*c[7]+m.b[10]*m.a[10,2]*m.a[2,1]*c[1]+m.b[10]*m.a[10,3]*m.a[3,1]*c[1]+m.b[10]*m.a[10,3]*m.a[3,2]*c[2]+m.b[10]*m.a[10,4]*m.a[4,1]*c[1]+m.b[10]*m.a[10,4]*m.a[4,2]*c[2]+m.b[10]*m.a[10,4]*m.a[4,3]*c[3]+m.b[10]*m.a[10,5]*m.a[5,1]*c[1]+m.b[10]*m.a[10,5]*m.a[5,2]*c[2]+m.b[10]*m.a[10,5]*m.a[5,3]*c[3]+m.b[10]*m.a[10,5]*m.a[5,4]*c[4]+m.b[10]*m.a[10,6]*m.a[6,1]*c[1]+m.b[10]*m.a[10,6]*m.a[6,2]*c[2]+m.b[10]*m.a[10,6]*m.a[6,3]*c[3]+m.b[10]*m.a[10,6]*m.a[6,4]*c[4]+m.b[10]*m.a[10,6]*m.a[6,5]*c[5]+m.b[10]*m.a[10,7]*m.a[7,1]*c[1]+m.b[10]*m.a[10,7]*m.a[7,2]*c[2]+m.b[10]*m.a[10,7]*m.a[7,3]*c[3]+m.b[10]*m.a[10,7]*m.a[7,4]*c[4]+m.b[10]*m.a[10,7]*m.a[7,5]*c[5]+m.b[10]*m.a[10,7]*m.a[7,6]*c[6]+m.b[10]*m.a[10,8]*m.a[8,1]*c[1]+m.b[10]*m.a[10,8]*m.a[8,2]*c[2]+m.b[10]*m.a[10,8]*m.a[8,3]*c[3]+m.b[10]*m.a[10,8]*m.a[8,4]*c[4]+m.b[10]*m.a[10,8]*m.a[8,5]*c[5]+m.b[10]*m.a[10,8]*m.a[8,6]*c[6]+m.b[10]*m.a[10,8]*m.a[8,7]*c[7]+m.b[10]*m.a[10,9]*m.a[9,1]*c[1]+m.b[10]*m.a[10,9]*m.a[9,2]*c[2]+m.b[10]*m.a[10,9]*m.a[9,3]*c[3]+m.b[10]*m.a[10,9]*m.a[9,4]*c[4]+m.b[10]*m.a[10,9]*m.a[9,5]*c[5]+m.b[10]*m.a[10,9]*m.a[9,6]*c[6]+m.b[10]*m.a[10,9]*m.a[9,7]*c[7]+m.b[10]*m.a[10,9]*m.a[9,8]*c[8]+m.b[11]*m.a[11,2]*m.a[2,1]*c[1]+m.b[11]*m.a[11,3]*m.a[3,1]*c[1]+m.b[11]*m.a[11,3]*m.a[3,2]*c[2]+m.b[11]*m.a[11,4]*m.a[4,1]*c[1]+m.b[11]*m.a[11,4]*m.a[4,2]*c[2]+m.b[11]*m.a[11,4]*m.a[4,3]*c[3]+m.b[11]*m.a[11,5]*m.a[5,1]*c[1]+m.b[11]*m.a[11,5]*m.a[5,2]*c[2]+m.b[11]*m.a[11,5]*m.a[5,3]*c[3]+m.b[11]*m.a[11,5]*m.a[5,4]*c[4]+m.b[11]*m.a[11,6]*m.a[6,1]*c[1]+m.b[11]*m.a[11,6]*m.a[6,2]*c[2]+m.b[11]*m.a[11,6]*m.a[6,3]*c[3]+m.b[11]*m.a[11,6]*m.a[6,4]*c[4]+m.b[11]*m.a[11,6]*m.a[6,5]*c[5]+m.b[11]*m.a[11,7]*m.a[7,1]*c[1]+m.b[11]*m.a[11,7]*m.a[7,2]*c[2]+m.b[11]*m.a[11,7]*m.a[7,3]*c[3]+m.b[11]*m.a[11,7]*m.a[7,4]*c[4]+m.b[11]*m.a[11,7]*m.a[7,5]*c[5]+m.b[11]*m.a[11,7]*m.a[7,6]*c[6]+m.b[11]*m.a[11,8]*m.a[8,1]*c[1]+m.b[11]*m.a[11,8]*m.a[8,2]*c[2]+m.b[11]*m.a[11,8]*m.a[8,3]*c[3]+m.b[11]*m.a[11,8]*m.a[8,4]*c[4]+m.b[11]*m.a[11,8]*m.a[8,5]*c[5]+m.b[11]*m.a[11,8]*m.a[8,6]*c[6]+m.b[11]*m.a[11,8]*m.a[8,7]*c[7]+m.b[11]*m.a[11,9]*m.a[9,1]*c[1]+m.b[11]*m.a[11,9]*m.a[9,2]*c[2]+m.b[11]*m.a[11,9]*m.a[9,3]*c[3]+m.b[11]*m.a[11,9]*m.a[9,4]*c[4]+m.b[11]*m.a[11,9]*m.a[9,5]*c[5]+m.b[11]*m.a[11,9]*m.a[9,6]*c[6]+m.b[11]*m.a[11,9]*m.a[9,7]*c[7]+m.b[11]*m.a[11,9]*m.a[9,8]*c[8]+m.b[11]*m.a[11,10]*m.a[10,1]*c[1]+m.b[11]*m.a[11,10]*m.a[10,2]*c[2]+m.b[11]*m.a[11,10]*m.a[10,3]*c[3]+m.b[11]*m.a[11,10]*m.a[10,4]*c[4]+m.b[11]*m.a[11,10]*m.a[10,5]*c[5]+m.b[11]*m.a[11,10]*m.a[10,6]*c[6]+m.b[11]*m.a[11,10]*m.a[10,7]*c[7]+m.b[11]*m.a[11,10]*m.a[10,8]*c[8]+m.b[11]*m.a[11,10]*m.a[10,9]*c[9]+m.b[12]*m.a[12,2]*m.a[2,1]*c[1]+m.b[12]*m.a[12,3]*m.a[3,1]*c[1]+m.b[12]*m.a[12,3]*m.a[3,2]*c[2]+m.b[12]*m.a[12,4]*m.a[4,1]*c[1]+m.b[12]*m.a[12,4]*m.a[4,2]*c[2]+m.b[12]*m.a[12,4]*m.a[4,3]*c[3]+m.b[12]*m.a[12,5]*m.a[5,1]*c[1]+m.b[12]*m.a[12,5]*m.a[5,2]*c[2]+m.b[12]*m.a[12,5]*m.a[5,3]*c[3]+m.b[12]*m.a[12,5]*m.a[5,4]*c[4]+m.b[12]*m.a[12,6]*m.a[6,1]*c[1]+m.b[12]*m.a[12,6]*m.a[6,2]*c[2]+m.b[12]*m.a[12,6]*m.a[6,3]*c[3]+m.b[12]*m.a[12,6]*m.a[6,4]*c[4]+m.b[12]*m.a[12,6]*m.a[6,5]*c[5]+m.b[12]*m.a[12,7]*m.a[7,1]*c[1]+m.b[12]*m.a[12,7]*m.a[7,2]*c[2]+m.b[12]*m.a[12,7]*m.a[7,3]*c[3]+m.b[12]*m.a[12,7]*m.a[7,4]*c[4]+m.b[12]*m.a[12,7]*m.a[7,5]*c[5]+m.b[12]*m.a[12,7]*m.a[7,6]*c[6]+m.b[12]*m.a[12,8]*m.a[8,1]*c[1]+m.b[12]*m.a[12,8]*m.a[8,2]*c[2]+m.b[12]*m.a[12,8]*m.a[8,3]*c[3]+m.b[12]*m.a[12,8]*m.a[8,4]*c[4]+m.b[12]*m.a[12,8]*m.a[8,5]*c[5]+m.b[12]*m.a[12,8]*m.a[8,6]*c[6]+m.b[12]*m.a[12,8]*m.a[8,7]*c[7]+m.b[12]*m.a[12,9]*m.a[9,1]*c[1]+m.b[12]*m.a[12,9]*m.a[9,2]*c[2]+m.b[12]*m.a[12,9]*m.a[9,3]*c[3]+m.b[12]*m.a[12,9]*m.a[9,4]*c[4]+m.b[12]*m.a[12,9]*m.a[9,5]*c[5]+m.b[12]*m.a[12,9]*m.a[9,6]*c[6]+m.b[12]*m.a[12,9]*m.a[9,7]*c[7]+m.b[12]*m.a[12,9]*m.a[9,8]*c[8]+m.b[12]*m.a[12,10]*m.a[10,1]*c[1]+m.b[12]*m.a[12,10]*m.a[10,2]*c[2]+m.b[12]*m.a[12,10]*m.a[10,3]*c[3]+m.b[12]*m.a[12,10]*m.a[10,4]*c[4]+m.b[12]*m.a[12,10]*m.a[10,5]*c[5]+m.b[12]*m.a[12,10]*m.a[10,6]*c[6]+m.b[12]*m.a[12,10]*m.a[10,7]*c[7]+m.b[12]*m.a[12,10]*m.a[10,8]*c[8]+m.b[12]*m.a[12,10]*m.a[10,9]*c[9]+m.b[12]*m.a[12,11]*m.a[11,1]*c[1]+m.b[12]*m.a[12,11]*m.a[11,2]*c[2]+m.b[12]*m.a[12,11]*m.a[11,3]*c[3]+m.b[12]*m.a[12,11]*m.a[11,4]*c[4]+m.b[12]*m.a[12,11]*m.a[11,5]*c[5]+m.b[12]*m.a[12,11]*m.a[11,6]*c[6]+m.b[12]*m.a[12,11]*m.a[11,7]*c[7]+m.b[12]*m.a[12,11]*m.a[11,8]*c[8]+m.b[12]*m.a[12,11]*m.a[11,9]*c[9]+m.b[12]*m.a[12,11]*m.a[11,10]*c[10]+m.b[13]*m.a[13,2]*m.a[2,1]*c[1]+m.b[13]*m.a[13,3]*m.a[3,1]*c[1]+m.b[13]*m.a[13,3]*m.a[3,2]*c[2]+m.b[13]*m.a[13,4]*m.a[4,1]*c[1]+m.b[13]*m.a[13,4]*m.a[4,2]*c[2]+m.b[13]*m.a[13,4]*m.a[4,3]*c[3]+m.b[13]*m.a[13,5]*m.a[5,1]*c[1]+m.b[13]*m.a[13,5]*m.a[5,2]*c[2]+m.b[13]*m.a[13,5]*m.a[5,3]*c[3]+m.b[13]*m.a[13,5]*m.a[5,4]*c[4]+m.b[13]*m.a[13,6]*m.a[6,1]*c[1]+m.b[13]*m.a[13,6]*m.a[6,2]*c[2]+m.b[13]*m.a[13,6]*m.a[6,3]*c[3]+m.b[13]*m.a[13,6]*m.a[6,4]*c[4]+m.b[13]*m.a[13,6]*m.a[6,5]*c[5]+m.b[13]*m.a[13,7]*m.a[7,1]*c[1]+m.b[13]*m.a[13,7]*m.a[7,2]*c[2]+m.b[13]*m.a[13,7]*m.a[7,3]*c[3]+m.b[13]*m.a[13,7]*m.a[7,4]*c[4]+m.b[13]*m.a[13,7]*m.a[7,5]*c[5]+m.b[13]*m.a[13,7]*m.a[7,6]*c[6]+m.b[13]*m.a[13,8]*m.a[8,1]*c[1]+m.b[13]*m.a[13,8]*m.a[8,2]*c[2]+m.b[13]*m.a[13,8]*m.a[8,3]*c[3]+m.b[13]*m.a[13,8]*m.a[8,4]*c[4]+m.b[13]*m.a[13,8]*m.a[8,5]*c[5]+m.b[13]*m.a[13,8]*m.a[8,6]*c[6]+m.b[13]*m.a[13,8]*m.a[8,7]*c[7]+m.b[13]*m.a[13,9]*m.a[9,1]*c[1]+m.b[13]*m.a[13,9]*m.a[9,2]*c[2]+m.b[13]*m.a[13,9]*m.a[9,3]*c[3]+m.b[13]*m.a[13,9]*m.a[9,4]*c[4]+m.b[13]*m.a[13,9]*m.a[9,5]*c[5]+m.b[13]*m.a[13,9]*m.a[9,6]*c[6]+m.b[13]*m.a[13,9]*m.a[9,7]*c[7]+m.b[13]*m.a[13,9]*m.a[9,8]*c[8]+m.b[13]*m.a[13,10]*m.a[10,1]*c[1]+m.b[13]*m.a[13,10]*m.a[10,2]*c[2]+m.b[13]*m.a[13,10]*m.a[10,3]*c[3]+m.b[13]*m.a[13,10]*m.a[10,4]*c[4]+m.b[13]*m.a[13,10]*m.a[10,5]*c[5]+m.b[13]*m.a[13,10]*m.a[10,6]*c[6]+m.b[13]*m.a[13,10]*m.a[10,7]*c[7]+m.b[13]*m.a[13,10]*m.a[10,8]*c[8]+m.b[13]*m.a[13,10]*m.a[10,9]*c[9]+m.b[13]*m.a[13,11]*m.a[11,1]*c[1]+m.b[13]*m.a[13,11]*m.a[11,2]*c[2]+m.b[13]*m.a[13,11]*m.a[11,3]*c[3]+m.b[13]*m.a[13,11]*m.a[11,4]*c[4]+m.b[13]*m.a[13,11]*m.a[11,5]*c[5]+m.b[13]*m.a[13,11]*m.a[11,6]*c[6]+m.b[13]*m.a[13,11]*m.a[11,7]*c[7]+m.b[13]*m.a[13,11]*m.a[11,8]*c[8]+m.b[13]*m.a[13,11]*m.a[11,9]*c[9]+m.b[13]*m.a[13,11]*m.a[11,10]*c[10]+m.b[13]*m.a[13,12]*m.a[12,1]*c[1]+m.b[13]*m.a[13,12]*m.a[12,2]*c[2]+m.b[13]*m.a[13,12]*m.a[12,3]*c[3]+m.b[13]*m.a[13,12]*m.a[12,4]*c[4]+m.b[13]*m.a[13,12]*m.a[12,5]*c[5]+m.b[13]*m.a[13,12]*m.a[12,6]*c[6]+m.b[13]*m.a[13,12]*m.a[12,7]*c[7]+m.b[13]*m.a[13,12]*m.a[12,8]*c[8]+m.b[13]*m.a[13,12]*m.a[12,9]*c[9]+m.b[13]*m.a[13,12]*m.a[12,10]*c[10]+m.b[13]*m.a[13,12]*m.a[12,11]*c[11]+m.b[14]*m.a[14,2]*m.a[2,1]*c[1]+m.b[14]*m.a[14,3]*m.a[3,1]*c[1]+m.b[14]*m.a[14,3]*m.a[3,2]*c[2]+m.b[14]*m.a[14,4]*m.a[4,1]*c[1]+m.b[14]*m.a[14,4]*m.a[4,2]*c[2]+m.b[14]*m.a[14,4]*m.a[4,3]*c[3]+m.b[14]*m.a[14,5]*m.a[5,1]*c[1]+m.b[14]*m.a[14,5]*m.a[5,2]*c[2]+m.b[14]*m.a[14,5]*m.a[5,3]*c[3]+m.b[14]*m.a[14,5]*m.a[5,4]*c[4]+m.b[14]*m.a[14,6]*m.a[6,1]*c[1]+m.b[14]*m.a[14,6]*m.a[6,2]*c[2]+m.b[14]*m.a[14,6]*m.a[6,3]*c[3]+m.b[14]*m.a[14,6]*m.a[6,4]*c[4]+m.b[14]*m.a[14,6]*m.a[6,5]*c[5]+m.b[14]*m.a[14,7]*m.a[7,1]*c[1]+m.b[14]*m.a[14,7]*m.a[7,2]*c[2]+m.b[14]*m.a[14,7]*m.a[7,3]*c[3]+m.b[14]*m.a[14,7]*m.a[7,4]*c[4]+m.b[14]*m.a[14,7]*m.a[7,5]*c[5]+m.b[14]*m.a[14,7]*m.a[7,6]*c[6]+m.b[14]*m.a[14,8]*m.a[8,1]*c[1]+m.b[14]*m.a[14,8]*m.a[8,2]*c[2]+m.b[14]*m.a[14,8]*m.a[8,3]*c[3]+m.b[14]*m.a[14,8]*m.a[8,4]*c[4]+m.b[14]*m.a[14,8]*m.a[8,5]*c[5]+m.b[14]*m.a[14,8]*m.a[8,6]*c[6]+m.b[14]*m.a[14,8]*m.a[8,7]*c[7]+m.b[14]*m.a[14,9]*m.a[9,1]*c[1]+m.b[14]*m.a[14,9]*m.a[9,2]*c[2]+m.b[14]*m.a[14,9]*m.a[9,3]*c[3]+m.b[14]*m.a[14,9]*m.a[9,4]*c[4]+m.b[14]*m.a[14,9]*m.a[9,5]*c[5]+m.b[14]*m.a[14,9]*m.a[9,6]*c[6]+m.b[14]*m.a[14,9]*m.a[9,7]*c[7]+m.b[14]*m.a[14,9]*m.a[9,8]*c[8]+m.b[14]*m.a[14,10]*m.a[10,1]*c[1]+m.b[14]*m.a[14,10]*m.a[10,2]*c[2]+m.b[14]*m.a[14,10]*m.a[10,3]*c[3]+m.b[14]*m.a[14,10]*m.a[10,4]*c[4]+m.b[14]*m.a[14,10]*m.a[10,5]*c[5]+m.b[14]*m.a[14,10]*m.a[10,6]*c[6]+m.b[14]*m.a[14,10]*m.a[10,7]*c[7]+m.b[14]*m.a[14,10]*m.a[10,8]*c[8]+m.b[14]*m.a[14,10]*m.a[10,9]*c[9]+m.b[14]*m.a[14,11]*m.a[11,1]*c[1]+m.b[14]*m.a[14,11]*m.a[11,2]*c[2]+m.b[14]*m.a[14,11]*m.a[11,3]*c[3]+m.b[14]*m.a[14,11]*m.a[11,4]*c[4]+m.b[14]*m.a[14,11]*m.a[11,5]*c[5]+m.b[14]*m.a[14,11]*m.a[11,6]*c[6]+m.b[14]*m.a[14,11]*m.a[11,7]*c[7]+m.b[14]*m.a[14,11]*m.a[11,8]*c[8]+m.b[14]*m.a[14,11]*m.a[11,9]*c[9]+m.b[14]*m.a[14,11]*m.a[11,10]*c[10]+m.b[14]*m.a[14,12]*m.a[12,1]*c[1]+m.b[14]*m.a[14,12]*m.a[12,2]*c[2]+m.b[14]*m.a[14,12]*m.a[12,3]*c[3]+m.b[14]*m.a[14,12]*m.a[12,4]*c[4]+m.b[14]*m.a[14,12]*m.a[12,5]*c[5]+m.b[14]*m.a[14,12]*m.a[12,6]*c[6]+m.b[14]*m.a[14,12]*m.a[12,7]*c[7]+m.b[14]*m.a[14,12]*m.a[12,8]*c[8]+m.b[14]*m.a[14,12]*m.a[12,9]*c[9]+m.b[14]*m.a[14,12]*m.a[12,10]*c[10]+m.b[14]*m.a[14,12]*m.a[12,11]*c[11]+m.b[14]*m.a[14,13]*m.a[13,1]*c[1]+m.b[14]*m.a[14,13]*m.a[13,2]*c[2]+m.b[14]*m.a[14,13]*m.a[13,3]*c[3]+m.b[14]*m.a[14,13]*m.a[13,4]*c[4]+m.b[14]*m.a[14,13]*m.a[13,5]*c[5]+m.b[14]*m.a[14,13]*m.a[13,6]*c[6]+m.b[14]*m.a[14,13]*m.a[13,7]*c[7]+m.b[14]*m.a[14,13]*m.a[13,8]*c[8]+m.b[14]*m.a[14,13]*m.a[13,9]*c[9]+m.b[14]*m.a[14,13]*m.a[13,10]*c[10]+m.b[14]*m.a[14,13]*m.a[13,11]*c[11]+m.b[14]*m.a[14,13]*m.a[13,12]*c[12]+m.b[15]*m.a[15,2]*m.a[2,1]*c[1]+m.b[15]*m.a[15,3]*m.a[3,1]*c[1]+m.b[15]*m.a[15,3]*m.a[3,2]*c[2]+m.b[15]*m.a[15,4]*m.a[4,1]*c[1]+m.b[15]*m.a[15,4]*m.a[4,2]*c[2]+m.b[15]*m.a[15,4]*m.a[4,3]*c[3]+m.b[15]*m.a[15,5]*m.a[5,1]*c[1]+m.b[15]*m.a[15,5]*m.a[5,2]*c[2]+m.b[15]*m.a[15,5]*m.a[5,3]*c[3]+m.b[15]*m.a[15,5]*m.a[5,4]*c[4]+m.b[15]*m.a[15,6]*m.a[6,1]*c[1]+m.b[15]*m.a[15,6]*m.a[6,2]*c[2]+m.b[15]*m.a[15,6]*m.a[6,3]*c[3]+m.b[15]*m.a[15,6]*m.a[6,4]*c[4]+m.b[15]*m.a[15,6]*m.a[6,5]*c[5]+m.b[15]*m.a[15,7]*m.a[7,1]*c[1]+m.b[15]*m.a[15,7]*m.a[7,2]*c[2]+m.b[15]*m.a[15,7]*m.a[7,3]*c[3]+m.b[15]*m.a[15,7]*m.a[7,4]*c[4]+m.b[15]*m.a[15,7]*m.a[7,5]*c[5]+m.b[15]*m.a[15,7]*m.a[7,6]*c[6]+m.b[15]*m.a[15,8]*m.a[8,1]*c[1]+m.b[15]*m.a[15,8]*m.a[8,2]*c[2]+m.b[15]*m.a[15,8]*m.a[8,3]*c[3]+m.b[15]*m.a[15,8]*m.a[8,4]*c[4]+m.b[15]*m.a[15,8]*m.a[8,5]*c[5]+m.b[15]*m.a[15,8]*m.a[8,6]*c[6]+m.b[15]*m.a[15,8]*m.a[8,7]*c[7]+m.b[15]*m.a[15,9]*m.a[9,1]*c[1]+m.b[15]*m.a[15,9]*m.a[9,2]*c[2]+m.b[15]*m.a[15,9]*m.a[9,3]*c[3]+m.b[15]*m.a[15,9]*m.a[9,4]*c[4]+m.b[15]*m.a[15,9]*m.a[9,5]*c[5]+m.b[15]*m.a[15,9]*m.a[9,6]*c[6]+m.b[15]*m.a[15,9]*m.a[9,7]*c[7]+m.b[15]*m.a[15,9]*m.a[9,8]*c[8]+m.b[15]*m.a[15,10]*m.a[10,1]*c[1]+m.b[15]*m.a[15,10]*m.a[10,2]*c[2]+m.b[15]*m.a[15,10]*m.a[10,3]*c[3]+m.b[15]*m.a[15,10]*m.a[10,4]*c[4]+m.b[15]*m.a[15,10]*m.a[10,5]*c[5]+m.b[15]*m.a[15,10]*m.a[10,6]*c[6]+m.b[15]*m.a[15,10]*m.a[10,7]*c[7]+m.b[15]*m.a[15,10]*m.a[10,8]*c[8]+m.b[15]*m.a[15,10]*m.a[10,9]*c[9]+m.b[15]*m.a[15,11]*m.a[11,1]*c[1]+m.b[15]*m.a[15,11]*m.a[11,2]*c[2]+m.b[15]*m.a[15,11]*m.a[11,3]*c[3]+m.b[15]*m.a[15,11]*m.a[11,4]*c[4]+m.b[15]*m.a[15,11]*m.a[11,5]*c[5]+m.b[15]*m.a[15,11]*m.a[11,6]*c[6]+m.b[15]*m.a[15,11]*m.a[11,7]*c[7]+m.b[15]*m.a[15,11]*m.a[11,8]*c[8]+m.b[15]*m.a[15,11]*m.a[11,9]*c[9]+m.b[15]*m.a[15,11]*m.a[11,10]*c[10]+m.b[15]*m.a[15,12]*m.a[12,1]*c[1]+m.b[15]*m.a[15,12]*m.a[12,2]*c[2]+m.b[15]*m.a[15,12]*m.a[12,3]*c[3]+m.b[15]*m.a[15,12]*m.a[12,4]*c[4]+m.b[15]*m.a[15,12]*m.a[12,5]*c[5]+m.b[15]*m.a[15,12]*m.a[12,6]*c[6]+m.b[15]*m.a[15,12]*m.a[12,7]*c[7]+m.b[15]*m.a[15,12]*m.a[12,8]*c[8]+m.b[15]*m.a[15,12]*m.a[12,9]*c[9]+m.b[15]*m.a[15,12]*m.a[12,10]*c[10]+m.b[15]*m.a[15,12]*m.a[12,11]*c[11]+m.b[15]*m.a[15,13]*m.a[13,1]*c[1]+m.b[15]*m.a[15,13]*m.a[13,2]*c[2]+m.b[15]*m.a[15,13]*m.a[13,3]*c[3]+m.b[15]*m.a[15,13]*m.a[13,4]*c[4]+m.b[15]*m.a[15,13]*m.a[13,5]*c[5]+m.b[15]*m.a[15,13]*m.a[13,6]*c[6]+m.b[15]*m.a[15,13]*m.a[13,7]*c[7]+m.b[15]*m.a[15,13]*m.a[13,8]*c[8]+m.b[15]*m.a[15,13]*m.a[13,9]*c[9]+m.b[15]*m.a[15,13]*m.a[13,10]*c[10]+m.b[15]*m.a[15,13]*m.a[13,11]*c[11]+m.b[15]*m.a[15,13]*m.a[13,12]*c[12]+m.b[15]*m.a[15,14]*m.a[14,1]*c[1]+m.b[15]*m.a[15,14]*m.a[14,2]*c[2]+m.b[15]*m.a[15,14]*m.a[14,3]*c[3]+m.b[15]*m.a[15,14]*m.a[14,4]*c[4]+m.b[15]*m.a[15,14]*m.a[14,5]*c[5]+m.b[15]*m.a[15,14]*m.a[14,6]*c[6]+m.b[15]*m.a[15,14]*m.a[14,7]*c[7]+m.b[15]*m.a[15,14]*m.a[14,8]*c[8]+m.b[15]*m.a[15,14]*m.a[14,9]*c[9]+m.b[15]*m.a[15,14]*m.a[14,10]*c[10]+m.b[15]*m.a[15,14]*m.a[14,11]*c[11]+m.b[15]*m.a[15,14]*m.a[14,12]*c[12]+m.b[15]*m.a[15,14]*m.a[14,13]*c[13] - (1/24)
    m.cons.add(y1 == 0 )
    m.cons.add(m.b[0] == m.b[1])
    m.cons.add(m.b[2] == m.b[1])
    m.cons.add(m.b[3] == m.b[1])
    m.cons.add(m.b[4] == m.b[1])
    m.cons.add(m.b[5] == m.b[1])
    m.cons.add(m.b[6] == m.b[1])
    m.cons.add(m.b[7] == m.b[1])
    #m.cons.add(m.b[8] == m.b[1])
    #m.cons.add(m.b[5] == m.a[2,1]**2)
 
   
    

#@title beta values { form-width: "100px" }

import numpy as np

data=np.array([
58.6541797842000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
1.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
0.50000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
0.166666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667,
0.041666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667,
0.0071849154521517355282317024156920385451420963577585214887176557969024167257745442301373054740876224,
0.00084158106959115128154963370362733819309862429010925098730105605001085577843601794969180423270290202,
0.000068644198049399319243540793509721822355678629849380897563717660112035417955533160190882128109809769,
4.0046982270542800501262550686500023006413459826466024539862702379518562524550740004475617513468536e-6,
1.70210679772636932238300318233994616504240328819532694428321378359410272554655187078812677529753515e-7,
5.3168095735087798640105832228671397136206973409466565391453256814310967819639282212573694895535897e-9,
1.21921914582208422817945588468800977605050266614889783816927720616912836571991663028322908220590411e-10,
2.02726315595571318800258905741043687139056504009610767488657621990319361287005588015056882588136873e-12,
2.37723157753774605171263935749525436011739345907744039018612185672514579560485148693189908827912392e-14,
1.86330977083129159365942538237637682996520930363206095611254106435484388267353014739842609411900501e-16,
8.7585588205225141798959725605331887162628631877211445208737323398801664983743986959153063247810753e-19,
1.86656749202421194921699690708646496298379636128946924630416992618329931984448468841233693522176513e-21
              ])
len(data)



#@title solve { form-width: "100px" }
#from __future__ import division
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition
#from pyomo.core.expr import current as EXPR
import pyomo.core.expr as EXPR
from pyomo.core.expr import value
import time
from random import uniform,seed,randrange
import sys

reloadmodel=False

#data = np.loadtxt('beta.dat')
betaval = data[0]
betdic = dict(enumerate(data[1:]))

minbet=min(data)
#csos23 bettol=1e-9*minbet
#csos23
bettol=1e-14


m = pyo.ConcreteModel()


##NB EDIT THIS BY HAND
m.order = pyo.Param(initialize=4)
m.macc = pyo.Param(initialize=4)
print('Order:',m.order.value,'Acc:',m.macc.value)



#csos23 m.s = pyo.Param(initialize=len(betdic)-1)
m.s = pyo.Param(initialize=m.order.value*m.macc.value)


initval=1./m.s.value   # csos23
#initval=0.5

lneg=1 # Set to 0 to disable negative coefficients in bounding and initialisation
lim=1
skip=1

print('lim:',lim,'skip:',skip,'initval:',initval,'lneg:',lneg)

def setlsrk(m, free_indices=[
    (1, 0),  # Stage 2 depends on Stage 1
    (2, 1),  # Stage 3 depends on Stage 2
    (3, 2),  # Stage 4 depends on Stage 3
    (4, 3),  # Stage 5 depends on Stage 4
    (5, 4),  # Stage 6 depends on Stage 5
    (6, 5),  # Stage 7 depends on Stage 6
    (7, 6),  # Stage 8 depends on Stage 7
    (8, 7),  # Stage 9 depends on Stage 8
    (9, 8),  # Stage 10 depends on Stage 9
    (10, 9), # Stage 11 depends on Stage 10
    (11, 10),# Stage 12 depends on Stage 11
    (12, 11),# Stage 13 depends on Stage 12
    (13, 12),# Stage 14 depends on Stage 13
    (14, 13), # Stage 15 depends on Stage 14
    (15,14),
    (16,15)
]):
    """
    This function sets up the constraints for the A-matrix and b-values.
    - All values in the lower triangular part of the A-matrix are tied to corresponding b-values.
    - The elements in the list `free_indices` are left as free parameters for optimization.
    - Modified to enforce a simple, sequential dependency for low-storage Runge-Kutta.
    """
    free_indices_set = set(free_indices)  # Convert list to set for fast lookup
    
    for j in range(m.s.value):  # Iterate over columns
        for i in range(j + 1, m.s.value):  # Iterate over rows below the diagonal (i > j)
            if (i, j) in free_indices_set:
                # This element is a free parameter; allow it to be optimized.
                continue
            else:
                # Enforce that all other a[i, j] == b[j]
                m.cons.add(m.a[i, j] == m.b[j])


def azeros(m):
    for i in m.rows:
        for j in range(i,m.s.value): #cols
            m.cons.add(m.a[i,j] == 0 )


m.rows= range(m.s.value)
m.cols= range(m.s.value)

m.a= pyo.Var( m.rows, m.cols, bounds=(-lneg*lim,lim))
m.b= pyo.Var( m.rows, bounds=(-lneg*lim,lim))
m.beta=pyo.Param(initialize=betaval)  #beta_R
m.bet = pyo.Param(range(m.s.value+1), initialize=betdic) #beta_j

print('beta_R:',m.beta.value)

m.cons = pyo.ConstraintList()

mycons(m)
#csos23
setlsrk(m)
#csos23?
#azeros(m) #Nones faster
mybets(m)

#m.pprint()


seedval = randrange(sys.maxsize)
#seedval=2842383511757470787
seed(seedval)
print('seedval:',seedval)



### solver = SolverFactory('ipopt',executable='/content/Ipopt-3.12.13/build/bin/ipopt')
solver = SolverFactory('ipopt',executable='ipopt')
solver.options['max_iter'] = 500000
solver.options['nlp_scaling_method'] = 'none'
solver.options['linear_solver'] = 'mumps'
solver.options['honor_original_bounds']='yes'

solver.options['bound_relax_factor']=bettol ## default 1e-8   https://kiwi.oden.utexas.edu/papers/Multi-output-multi-fidelity-MLBLUE-Croci-Wright-Willcox.pdf

solver.options['tol'] = bettol # default 1e-8
solver.options['dual_inf_tol']=1e8*bettol #default 1
solver.options['constr_viol_tol']=1e4*bettol #default 1e-4
solver.options['compl_inf_tol']=1e4*bettol #default 1e-4

solver.options['acceptable_tol'] = bettol # default 1e-6
solver.options['acceptable_dual_inf_tol']=1e8*bettol #default 1e10
solver.options['acceptable_constr_viol_tol']=1e4*bettol #default 1e-2
solver.options['acceptable_compl_inf_tol']=1e4*bettol #default 1e-2



niter=1

#solver = SolverFactory('multistart')
#niter=1
#nitermulti=10

#solver = SolverFactory('couenne')
#niter=1


if reloadmodel:
    with open('m.pkl', mode='rb') as file: #reload model
        m = cloudpickle.load(file)
        print('Reload')
else:
    objvalbest=1e10
    for iter in range(niter):

        start = time.time()

        for i in range(1,m.s.value):
            for j in range(i):
                #csos23
                m.a[i,j].value=uniform(-lneg*initval,initval)
                #m.a[i,j].value=initval
        for j in m.cols:
            #csos23
            m.b[i].value=uniform(-lneg*initval,initval)
            #m.b[i].value=initval   ## sum b=1 !



        try:

            results=solver.solve(m,tee=True)
            #results=solver.solve(m,tee=False)

            #results=solver.solve(m,solver='ipopt',strategy='rand_distributed',iterations=nitermulti)
            if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                end = time.time()
                objval=m.obj()
                print(iter,'Time:',end - start,'Objective:',objval)
                if objval < objvalbest:
                    objvalbest=objval
                    mbest=m.clone()
                    resultsbest=results
            elif (results.solver.termination_condition == TerminationCondition.infeasible):
                print(iter,'Infeasible.')
            else:
                print(iter,'Failed. Solver Status:',  result.solver.status)
        except:
            print(iter,'Ipopt error, moving on.')

        #m.display()
        #results.write()


    m=mbest.clone()  # revert to best instance
    results=resultsbest


    import cloudpickle  # write model to file
    with open('m.pkl', mode='wb') as file:
        cloudpickle.dump(m, file)


objval=m.obj()
print('Final objective value:',objval)


#@title tableau { form-width: "100px" }

a =  []
b= []
print('A\n')
for i in range(1,m.s.value):
    for j in range(i):
        c =  float(m.a[i,j].value)
        a.append(c)
#print("\n")


print("<=================================================>")
print("B\n")
for j in m.cols:
    c=float(m.b[j].value)
    b.append(c)
 

print(a)
print(b)
x  =  m.a[15,14].value + m.a[1,0].value + m.b[4].value

print(m.b[3].value)
print("\n")
print(x)
import sympy as sym
s=m.s.value
zval=-betaval
sym.init_printing()
x,y = sym.symbols('x y',real=True)
#z=x+sym.I*y
z = sym.symbols('z',complex=True)
def af(i,j):
    if j <i:
        return m.a[i,j].value
    else:
        return 0
def bf(i,j):
    return m.b[i].value
def ef(i,j):
    return 1

amat = sym.Matrix(s,s,af)
bvec = sym.Matrix(s,1,bf)
evec=sym.Matrix(s,1,ef)
bmat=sym.eye(s)-z*amat
binv=bmat.inv()
Qvecz=z*bvec.T*binv
Qvec=Qvecz.subs(z,x+sym.I*y)
Rz=sym.simplify(1+(Qvec*evec)[0])
R=Rz.subs(z,x+sym.I*y)
Rstab = sym.lambdify( [x,y], R, "numpy" )
Qs=[]
for i in range(s):
    Qs.append(sym.lambdify( [x,y], Qvec[i], "numpy" ))


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
f=plt.figure(figsize=(16, s*4))
#f, axes = plt.subplots(s, 1, figsize=(8, s*4), sharex=True, sharey=True)
x = np.linspace(-1.00*betaval, 0.0*betaval, 100)
y = np.linspace(-0.2*betaval, 0.2*betaval, 100)
X, Y = np.meshgrid(x, y)
ZR=np.abs(Rstab(X,Y))
for i in range(s):
    plt.subplot(s,1,i+1)
    Z=np.log10(np.abs(Qs[i](X,Y)))
    c1=plt.contourf(X,Y,Z,cmap='RdBu')
    c2=plt.contour(X,Y,ZR,colors='black',levels=[1])
    plt.colorbar(c1)
f.tight_layout()
plt.savefig('Qstabs.png', bbox_inches='tight')
