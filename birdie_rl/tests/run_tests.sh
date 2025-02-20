#!/bin/bash

# Clear the terminal screen
clear

printf "\n\n"
echo "*** Running objectives tests ***"

printf "\n\n**********\n\n"
echo "Autoencoding:"
python -m objectives.autoencoding

printf "\n\n**********\n\n"
echo "Deshuffling:"
python -m objectives.deshuffling

printf "\n\n**********\n\n"
echo "Infilling:"
python -m objectives.infilling

printf "\n\n**********\n\n"
echo "Next token prediction:"
python -m objectives.next_token_prediction

printf "\n\n**********\n\n"
echo "Prefix language modeling:"
python -m objectives.prefix_language_modeling

printf "\n\n**********\n\n"
echo "Selective copying:"
python -m objectives.selective_copying

printf "\n\n\n\n"
python -c "print(flush=True)"

echo "test results:"
echo "sam@sam:~$ ./tests/run_tests.sh"

# Run the test suite with coverage tracking
coverage run --source=. -m unittest discover -s tests
echo ""
coverage report -m
echo ""
# Uncomment to generate HTML coverage
# coverage html
# echo "Open file://$(pwd)/htmlcov/index.html to see the coverage report."
