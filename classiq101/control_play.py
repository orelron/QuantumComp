from classiq import *

@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(n_ctrl_qbits, cntrl)
    allocate(1, target)
    control(ctrl=cntrl, stmt_block=lambda: X(target))

for n_ctrl_qbits in range(5,21):
    quantum_model = create_model(main)
    quantum_model_with_constraints = set_constraints(
        quantum_model, Constraints(optimization_parameter="width")
    )
    quantum_program = synthesize(quantum_model_with_constraints)
    width = QuantumProgram.from_qprog(quantum_program).data.width
    depth = QuantumProgram.from_qprog(quantum_program).transpiled_circuit.depth
    print(f"{n_ctrl_qbits}-control bits: {width=}, {depth=}")

show(quantum_program)

quantum_model_with_constraints = set_constraints(
    quantum_model, Constraints(optimization_parameter="depth")
)
quantum_program = synthesize(quantum_model_with_constraints)
show(quantum_program)
