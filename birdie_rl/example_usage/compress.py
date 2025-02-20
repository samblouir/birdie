#!/usr/bin/env python3
import sys

def stream_compress():
    current_line = None
    count = 0
    try:
        for line in sys.stdin:
            # Remove the newline at the end
            line = line.rstrip("\n")
            if current_line is None:
                current_line = line
                count = 1
            elif line == current_line:
                count += 1
            else:
                # Output the previous group immediately
                sys.stdout.write(current_line + "\n")
                if count > 1:
                    sys.stdout.write(f"  [Last line repeated {count - 1} times...]\n")
                sys.stdout.flush()
                current_line = line
                count = 1
        # When the stream ends, flush any remaining lines
        if current_line is not None:
            sys.stdout.write(current_line + "\n")
            if count > 1:
                sys.stdout.write(f"  [Last line repeated {count - 1} times...]\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        # Allow clean termination on Ctrl+C
        pass

if __name__ == '__main__':
    stream_compress()
