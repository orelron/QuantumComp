from classiq import *


@qfunc
def main(x: Output[QNum], y: Output[QNum]):

    allocate(4, x)
    hadamard_transform(x)  # creates a uniform superposition
    y |= x**2 + 1

quantum_model = create_model(main)
quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="width", max_depth=500)
)
quantum_program = synthesize(quantum_model_with_constraints)
show(quantum_program)
