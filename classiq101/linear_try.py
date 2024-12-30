from classiq import *
import numpy as np
import matplotlib.pyplot as plt

# If you have not yet authenticated please uncomment the line below
# authenticate()

@qfunc
def main(x:Output[QNum], y: Output[QNum]) -> None:
    K = 4
    allocate_num(num_qubits=K, is_signed=False, fraction_digits=K, out=x)
    allocate_num(num_qubits=K, is_signed=False, fraction_digits=K, out=y)
    hadamard_transform(x)
    small = QBit('small')
    medium = QBit('medium')
    big = QBit('big')
    small |= (x>0.1)
    big   |= (x>0.9)
    medium|= (x>0.4) & (x<0.6)
    # ctrl will be 0, 1, 2, 3, ... according to small, big, medium, ...
    ctrl = QNum('ctrl')
    ctrl |= small + 2*medium + 4*big
    control_logic(a=[1,2], b=[1,2], controller=ctrl, x=x, y=y)

@qfunc
def linear_func(a: CInt, b: CInt, x:QNum, y: Output[QNum]):
    y |= a*x+b

@qfunc
def inplace_linear_func(a: CInt, b: CInt, x: QNum, y: QNum):
    tmp = QNum('tmp')
    within_apply(within=lambda: linear_func(a,b,x,tmp), apply=lambda: inplace_xor(tmp,y))

@qfunc
def control_logic(a: CArray[int], b: CArray[int], controller: QNum, x: QNum, y: QNum):
    repeat(count=a.len,         
        iteration=lambda i: control(controller==i, lambda: inplace_linear_func(a[i],b[i],x,y)))   

quantum_model = create_model(main)

# width as the optimization parameter
quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="depth", max_width=25)
)
quantum_program =  synthesize(quantum_model_with_constraints)
depth = int(QuantumProgram.from_qprog(quantum_program).transpiled_circuit.depth)
print(f"depth={depth}")
show(quantum_program)
