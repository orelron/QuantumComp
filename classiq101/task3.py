from classiq import *
import numpy as np
import matplotlib.pyplot as plt

# If you have not yet authenticated please uncomment the line below
# authenticate()

@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(20, cntrl)
    allocate(1, target)
    hadamard_transform(cntrl)
    control(ctrl=cntrl, stmt_block=lambda: X(target))

width = np.zeros(31-22, dtype=int)
depth = np.zeros(31-22, dtype=int)
for i_width, max_width in enumerate(range(22,31)):
    quantum_model = create_model(main)
    quantum_model_with_constraints = set_constraints(
        quantum_model, Constraints(optimization_parameter="depth", max_width=max_width)
    )
    quantum_program = synthesize(quantum_model_with_constraints)

    # Save width and depth
    width[i_width] = int(QuantumProgram.from_qprog(quantum_program).data.width)
    depth[i_width] = int(QuantumProgram.from_qprog(quantum_program).transpiled_circuit.depth)
    assert width[i_width]==max_width, "Incorrect width!" 
    # show(quantum_program)
    print(f"width={max_width}, depth={depth[i_width]}")

# Plot
plt.plot(width, depth, marker='x', linestyle='-', color='r', label="Traspiled Depth")
plt.xlabel("Width")
plt.ylabel("Depth")
plt.title("Depth vs. Width")
plt.show(block=True)
