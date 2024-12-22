from classiq import *

# If you have not yet authenticated please uncomment the line below
# authenticate()

@qfunc
def flip_msb(reg: QArray):
    X(reg[reg.len - 1])

@qfunc
def flip_lsb(reg: QArray):
    X(reg[0])

@qfunc
def main(indicator: Output[QBit]):

    x = QNum("x")
    allocate(4, x)
    flip_msb(x)
    flip_lsb(x)

    indicator |= x == 9

quantum_program = synthesize(create_model(main))
show(quantum_program)
