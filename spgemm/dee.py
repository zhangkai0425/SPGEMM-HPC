import os
with open('real.txt', 'r') as real_file, open('fake.txt', 'r') as fake_file:
    for line_number, (real_line, fake_line) in enumerate(zip(real_file, fake_file), start=1):
        real_value = int(real_line.strip())
        fake_value = int(fake_line.strip())

        if fake_value < real_value:
            print(f"Line {line_number}: Real value = {real_value}, Fake value = {fake_value}")
