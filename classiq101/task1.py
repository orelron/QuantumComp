from classiq import *

# If you have not yet authenticated please uncomment the line below
# authenticate()

@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(5, cntrl)
    allocate(1, target)
    control(ctrl=cntrl, stmt_block=lambda: X(target))

quantum_model = create_model(main)

quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="width")
)
quantum_program = synthesize(quantum_model_with_constraints)
show(quantum_program)

quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="depth")
)
quantum_program = synthesize(quantum_model_with_constraints)
show(quantum_program)

quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="depth", max_width=7)
)
quantum_program = synthesize(quantum_model_with_constraints)
show(quantum_program)
