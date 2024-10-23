import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto


class GateType(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()
    NAND = auto()
    NOR = auto()
    XNOR = auto()
    BUF = auto()
    NOT = auto()


@dataclass
class BitRange:
    msb: int
    lsb: int

    @property
    def width(self) -> int:
        return self.msb - self.lsb + 1


@dataclass
class Wire:
    name: str
    bit_range: Optional[BitRange] = None
    is_input: bool = False
    is_output: bool = False
    current_value: int = 0

    @property
    def is_vector(self) -> bool:
        return self.bit_range is not None

    @property
    def width(self) -> int:
        return self.bit_range.width if self.is_vector else 1


@dataclass
class BitAccess:
    wire_name: str
    bit_index: Optional[int] = None

    @classmethod
    def parse(cls, signal: str) -> 'BitAccess':
        """Parse a signal name that might include bit access."""
        match = re.match(r'(\w+)(?:\[(\d+)\])?', signal.strip())
        if not match:
            raise ValueError(f"Invalid signal format: {signal}")
        wire_name = match.group(1)
        bit_index = int(match.group(2)) if match.group(2) else None
        return cls(wire_name, bit_index)


@dataclass
class Gate:
    gate_type: GateType
    name: str
    output: BitAccess
    inputs: List[BitAccess]
    delay: int  # delay in ps


class VerilogParser:
    def _init_(self):
        self.module_name: str = ""
        self.inputs: Set[str] = set()
        self.outputs: Set[str] = set()
        self.wires: Dict[str, Wire] = {}
        self.gates: List[Gate] = []

        self.gate_map = {
            'and': GateType.AND,
            'or': GateType.OR,
            'xor': GateType.XOR,
            'nand': GateType.NAND,
            'nor': GateType.NOR,
            'xnor': GateType.XNOR,
            'buf': GateType.BUF,
            'not': GateType.NOT
        }

    def parse_file(self, filename: str) -> bool:
        """Parse a Verilog file and extract circuit information."""
        try:
            with open(filename, 'r') as file:
                content = file.read()

            # Remove comments
            content = re.sub(r'//.*?\n', '\n', content)  # Remove single-line comments
            content = re.sub(r'/\.?\*/', '', content, flags=re.DOTALL)  # Remove multi-line comments

            # Remove line continuations and merge lines
            content = re.sub(r'\\\s*\n\s*', ' ', content)

            # Normalize whitespace while preserving newlines
            content = '\n'.join(line.strip() for line in content.splitlines())

            # Extract module information
            self._parse_module(content)

            # Extract port declarations
            self._parse_ports(content)

            # Extract wire declarations
            self._parse_wires(content)

            # Extract gate instantiations
            self._parse_gates(content)

            # Verify circuit consistency
            self._verify_circuit()

            return True

        except Exception as e:
            print(f"Error parsing Verilog file: {str(e)}")
            return False

    def _parse_module(self, content: str) -> None:
        """Extract module name and ports with vector information."""
        # Match module declaration including vector information
        module_pattern = r'module\s+(\w+)\s*\(([\s\S]*?)\);'
        match = re.search(module_pattern, content)
        if not match:
            raise ValueError("No module found in Verilog file")

        self.module_name = match.group(1)

        # Initialize storage for port information
        self.port_ranges = {}

        # First, find all input/output declarations to know port types
        input_decl = r'input\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)'
        output_decl = r'output\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)'

        # Process input declarations
        for match in re.finditer(input_decl, content):
            msb, lsb, name = match.groups()
            if msb is not None and lsb is not None:
                self.port_ranges[name] = BitRange(int(msb), int(lsb))
            self.inputs.add(name)
            print(f"Debug: Found input port {name}")

        # Process output declarations
        for match in re.finditer(output_decl, content):
            msb, lsb, name = match.groups()
            if msb is not None and lsb is not None:
                self.port_ranges[name] = BitRange(int(msb), int(lsb))
            self.outputs.add(name)
            print(f"Debug: Found output port {name}")

        # Look for vector declarations in the module header
        vector_in_header = r'\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*(\w+)'
        header_vectors = re.finditer(vector_in_header, content)
        for match in header_vectors:
            msb, lsb, name = match.groups()
            self.port_ranges[name] = BitRange(int(msb), int(lsb))
            print(f"Debug: Found header vector {name}[{msb}:{lsb}]")

    def _parse_ports(self, content: str) -> None:
        """Parse input and output port declarations."""
        # Process inputs
        input_pattern = r'input\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*[,;]'
        for match in re.finditer(input_pattern, content):
            msb, lsb, name = match.groups()
            bit_range = None

            # Check for range in declaration or port_ranges
            if msb is not None and lsb is not None:
                bit_range = BitRange(int(msb), int(lsb))
            elif name in self.port_ranges:
                bit_range = self.port_ranges[name]

            # Ensure it's in the inputs set
            self.inputs.add(name)
            # Create the wire if it doesn't exist
            if name not in self.wires:
                self.wires[name] = Wire(name, bit_range, is_input=True)
                print(f"Debug: Created input wire {name} with range {bit_range}")

        # Process outputs
        output_pattern = r'output\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*[,;]'
        for match in re.finditer(output_pattern, content):
            msb, lsb, name = match.groups()
            bit_range = None

            # Check for range in declaration or port_ranges
            if msb is not None and lsb is not None:
                bit_range = BitRange(int(msb), int(lsb))
            elif name in self.port_ranges:
                bit_range = self.port_ranges[name]

            # Ensure it's in the outputs set
            self.outputs.add(name)
            # Create the wire if it doesn't exist
            if name not in self.wires:
                self.wires[name] = Wire(name, bit_range, is_output=True)
                print(f"Debug: Created output wire {name} with range {bit_range}")

    def set_bit_value(self, signal: BitAccess, value: int) -> None:
        """Set the value of a specific bit or entire signal."""
        wire = self.wires[signal.wire_name]
        if signal.bit_index is not None:
            if not wire.is_vector:
                raise ValueError(f"Attempting to set bit {signal.bit_index} of non-vector signal {signal.wire_name}")
            # Clear the bit
            mask = ~(1 << signal.bit_index)
            wire.current_value &= mask
            # Set the new bit value
            if value:
                wire.current_value |= (1 << signal.bit_index)
        else:
            max_value = (1 << wire.width) - 1
            wire.current_value = value & max_value

    def print_signal_values(self) -> None:
        """Print current values of all signals, including individual bits for vectors."""
        print("\nCurrent Signal Values:")
        for name, wire in sorted(self.wires.items()):
            if wire.is_vector:
                bits = []
                for i in range(wire.bit_range.msb, wire.bit_range.lsb - 1, -1):
                    bit_value = self.get_bit_value(BitAccess(name, i))
                    bits.append(str(bit_value))
                print(
                    f"  {name}[{wire.bit_range.msb}:{wire.bit_range.lsb}] = {wire.current_value} (binary: {''.join(bits)})")
            else:
                print(f"  {name} = {wire.current_value}")

    def _parse_bit_range(self, declaration: str) -> Tuple[str, Optional[BitRange]]:
        """Parse a signal declaration that might include a bit range."""
        # Handle both "signal[3:0]" and "[3:0] signal" formats
        match = re.match(r'(?:\s*\[(\d+):(\d+)\])?\s*(\w+)|\s*(\w+)\s*\[(\d+):(\d+)\]', declaration.strip())
        if not match:
            raise ValueError(f"Invalid signal declaration: {declaration}")

        # Check which format matched
        if match.group(3):  # Format: [3:0] signal
            name = match.group(3)
            if match.group(1) and match.group(2):
                msb = int(match.group(1))
                lsb = int(match.group(2))
                return name, BitRange(msb, lsb)
        else:  # Format: signal[3:0]
            name = match.group(4)
            if match.group(5) and match.group(6):
                msb = int(match.group(5))
                lsb = int(match.group(6))
                return name, BitRange(msb, lsb)

        return name, None

    def _parse_ports(self, content: str) -> None:
        """Parse input and output port declarations with bit ranges."""
        # First, find the module declaration to get any bit ranges
        module_pattern = r'module\s+\w+\s*\((.*?)\);'
        module_match = re.search(module_pattern, content, re.DOTALL)
        if not module_match:
            raise ValueError("Module declaration not found")

        port_list = module_match.group(1)
        # Store port declarations for reference
        ports_with_ranges = {}

        # Extract bit ranges from port declarations in module header
        port_range_pattern = r'(?:\[(\d+):(\d+)\])?\s*(\w+)'
        matches = re.finditer(port_range_pattern, port_list)
        for match in matches:
            msb, lsb, port_name = match.groups()
            if msb is not None and lsb is not None:
                ports_with_ranges[port_name] = BitRange(int(msb), int(lsb))

        # Find input declarations
        input_pattern = r'input\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*;'
        for match in re.finditer(input_pattern, content):
            msb, lsb, name = match.groups()

            # Use either the range from module declaration or from input declaration
            if msb is not None and lsb is not None:
                bit_range = BitRange(int(msb), int(lsb))
            else:
                bit_range = ports_with_ranges.get(name)

            self.inputs.add(name)
            self.wires[name] = Wire(name, bit_range, is_input=True)
            print(f"Debug: Added input wire {name} with range {bit_range}")  # Debug print

        # Find output declarations
        output_pattern = r'output\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*;'
        for match in re.finditer(output_pattern, content):
            msb, lsb, name = match.groups()

            # Use either the range from module declaration or from output declaration
            if msb is not None and lsb is not None:
                bit_range = BitRange(int(msb), int(lsb))
            else:
                bit_range = ports_with_ranges.get(name)

            self.outputs.add(name)
            self.wires[name] = Wire(name, bit_range, is_output=True)
            print(f"Debug: Added output wire {name} with range {bit_range}")  # Debug print

    def _parse_wires(self, content: str) -> None:
        """Parse wire declarations."""
        wire_pattern = r'wire\s+(?:\[(\d+):(\d+)\]\s*)?(\w+)\s*[,;]'
        for match in re.finditer(wire_pattern, content):
            msb, lsb, name = match.groups()
            bit_range = None

            if msb is not None and lsb is not None:
                bit_range = BitRange(int(msb), int(lsb))
            elif name in self.port_ranges:
                bit_range = self.port_ranges[name]

            if name not in self.wires:  # Don't overwrite ports
                self.wires[name] = Wire(name, bit_range)
                print(f"Debug: Created wire {name} with range {bit_range}")

    def _add_implicit_wire(self, wire_name: str) -> None:
        """Add a wire that wasn't explicitly declared."""
        if wire_name not in self.wires and wire_name.isalnum():
            self.wires[wire_name] = Wire(wire_name)

    def _parse_connection(self, conn: str) -> BitAccess:
        """Parse a connection string and handle bit access."""
        conn = conn.strip()

        # Handle constant values
        if conn == '0' or conn == '1':
            constant_wire = f"CONST_{conn}"
            if constant_wire not in self.wires:
                self.wires[constant_wire] = Wire(constant_wire)
                self.wires[constant_wire].current_value = int(conn)
            return BitAccess(constant_wire)

        # Parse bit access
        match = re.match(r'(\w+)(?:\[(\d+)\])?', conn)
        if not match:
            raise ValueError(f"Invalid connection format: {conn}")

        wire_name = match.group(1)
        bit_index = int(match.group(2)) if match.group(2) else None

        # Create wire if it doesn't exist, using stored vector information
        if wire_name not in self.wires:
            if wire_name in self.port_ranges:
                bit_range = self.port_ranges[wire_name]
                self.wires[wire_name] = Wire(wire_name, bit_range)
                print(f"Debug: Created wire {wire_name} with range {bit_range}")
            elif wire_name in self.outputs or wire_name in self.inputs:
                bit_range = self.port_ranges.get(wire_name)
                self.wires[wire_name] = Wire(wire_name, bit_range)
                print(f"Debug: Created port wire {wire_name} with range {bit_range}")
            else:
                self._add_implicit_wire(wire_name)
                print(f"Debug: Created implicit wire {wire_name} without range")

        return BitAccess(wire_name, bit_index)

    def _add_implicit_wire(self, wire_name: str) -> None:
        """Add a wire that wasn't explicitly declared."""
        if wire_name not in self.wires:
            # Check if we have vector information for this wire
            bit_range = self.port_ranges.get(wire_name)
            self.wires[wire_name] = Wire(wire_name, bit_range)
    def _parse_gates(self, content: str) -> None:
        """Parse gate instantiations with bit-specific connections."""
        gate_pattern = r'(and|or|xor|nand|nor|xnor|buf|not)\s*(?:#(\d+))?\s*(\w+)\s*\(([\s\S]*?)\);'

        for match in re.finditer(gate_pattern, content):
            gate_type = match.group(1)
            delay = int(match.group(2)) if match.group(2) else 0
            gate_name = match.group(3)
            connections = [conn.strip() for conn in match.group(4).split(',')]

            # Parse and normalize all connections
            output = self._parse_connection(connections[0])
            inputs = [self._parse_connection(conn) for conn in connections[1:]]

            # Add implicit wires for base signals
            self._add_implicit_wire(output.wire_name)
            for input_access in inputs:
                self._add_implicit_wire(input_access.wire_name)

            # Create gate instance
            gate = Gate(
                gate_type=self.gate_map[gate_type],
                name=gate_name,
                output=output,
                inputs=inputs,
                delay=delay
            )

            self.gates.append(gate)

    def _verify_circuit(self) -> None:
        """Verify circuit consistency and completeness."""
        for gate in self.gates:
            # Verify output wire exists and bit access is valid
            out_wire = self.wires.get(gate.output.wire_name)
            if out_wire is None:
                raise ValueError(f"Gate {gate.name} output wire {gate.output.wire_name} not found")

            # Validate output bit access
            if gate.output.bit_index is not None:
                if not out_wire.is_vector:
                    raise ValueError(
                        f"Gate {gate.name}: Attempting to access bit {gate.output.bit_index} "
                        f"of non-vector signal {gate.output.wire_name}")
                if gate.output.bit_index > out_wire.bit_range.msb or gate.output.bit_index < out_wire.bit_range.lsb:
                    raise ValueError(
                        f"Gate {gate.name}: Bit index {gate.output.bit_index} out of range for "
                        f"{gate.output.wire_name}[{out_wire.bit_range.msb}:{out_wire.bit_range.lsb}]")

            # Verify input wires exist and bit access is valid
            for input_access in gate.inputs:
                # Skip validation for constant values
                if input_access.wire_name.startswith('CONST_'):
                    continue

                in_wire = self.wires.get(input_access.wire_name)
                if in_wire is None:
                    raise ValueError(f"Gate {gate.name} input wire {input_access.wire_name} not found")

                # Validate input bit access
                if input_access.bit_index is not None:
                    if not in_wire.is_vector:
                        raise ValueError(
                            f"Gate {gate.name}: Attempting to access bit {input_access.bit_index} "
                            f"of non-vector signal {input_access.wire_name}")
                    if input_access.bit_index > in_wire.bit_range.msb or input_access.bit_index < in_wire.bit_range.lsb:
                        raise ValueError(
                            f"Gate {gate.name}: Bit index {input_access.bit_index} out of range for "
                            f"{input_access.wire_name}[{in_wire.bit_range.msb}:{in_wire.bit_range.lsb}]")

            # Verify gate input count
            valid_inputs = {
                GateType.NOT: 1,
                GateType.BUF: 1
            }
            if gate.gate_type in valid_inputs and len(gate.inputs) != valid_inputs[gate.gate_type]:
                raise ValueError(
                    f"Gate {gate.name} of type {gate.gate_type.name} requires exactly "
                    f"{valid_inputs[gate.gate_type]} input(s), but got {len(gate.inputs)}")
    def print_circuit_summary(self) -> None:
        """Print a summary of the parsed circuit."""
        print(f"\nCircuit Summary for module '{self.module_name}':")
        print("\nInputs:")
        for name in sorted(self.inputs):
            wire = self.wires.get(name)
            if wire and wire.is_vector:
                print(f"  {name}[{wire.bit_range.msb}:{wire.bit_range.lsb}]")
            else:
                print(f"  {name}")

        print("\nOutputs:")
        for name in sorted(self.outputs):
            wire = self.wires.get(name)
            if wire and wire.is_vector:
                print(f"  {name}[{wire.bit_range.msb}:{wire.bit_range.lsb}]")
            else:
                print(f"  {name}")

        print("\nGates:")
        for gate in self.gates:
            print(f"  {gate.gate_type.name} gate '{gate.name}':")
            out_bit = f"[{gate.output.bit_index}]" if gate.output.bit_index is not None else ""
            print(f"    Output: {gate.output.wire_name}{out_bit}")
            print("    Inputs:", end=" ")
            inputs = []
            for inp in gate.inputs:
                bit = f"[{inp.bit_index}]" if inp.bit_index is not None else ""
                inputs.append(f"{inp.wire_name}{bit}")
            print(", ".join(inputs))
            print(f"    Delay: {gate.delay}ps")


# Test the parser with a sample circuit
if _name_ == "_main_":
    test_verilog = """
    module test_circuit(
    A, B, Y  // Port list
);
    // Port declarations with vectors
    input [3:0] A;
    input [3:0] B;
    output [3:0] Y;

    // Internal wires
    wire w1;

    // Gates
    and #1000 G1(w1, A[1], B[2]);
    or #1000 G2(Y[0], w1, A[0]);

endmodule
    """

    with open("test_vector.v", "w") as f:
        f.write(test_verilog)

    parser = VerilogParser()
    if parser.parse_file("example-verilog.v"):
        parser.print_circuit_summary()
