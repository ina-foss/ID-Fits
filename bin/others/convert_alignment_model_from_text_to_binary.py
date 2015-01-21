import struct
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converts a model from text format to binary format.")
    parser.add_argument("input_text_file", help="Input model file in text format to convert.")
    parser.add_argument("output_file", help="Output model file in binary format.")
    args = parser.parse_args()


    input_file = open(args.input_text_file, "r")
    output_file = open(args.output_file, "wb")

    T, N, D, L = [int(x) for x in input_file.readline().split()]
    output_file.write(struct.pack("<4I", T, N, D, L))

    mean_shape = map(float, input_file.readline().split())
    output_file.write(struct.pack("<%if" % len(mean_shape), *mean_shape))

    for _ in range(T):
        for _ in range(L):
            for _ in range(N):
                for _ in range(2**(D+1)-1):
                    line = input_file.readline().split()
                    d, p = map(int, line[:2])
                    indexes = map(float, line[2:])
                    output_file.write(struct.pack("<2I4f", d, p, *indexes))

            l = int(input_file.readline())
            output_file.write(struct.pack("<I", l))

            for _ in range(N*(2**D)):
                increment = map(float, input_file.readline().split())
                output_file.write(struct.pack("<%if" % len(increment), *increment))
