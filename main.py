from ExperimentalPipeline1 import main

# Specific rows correspond to configurations defined in the CSV files (See ExperimentList folder).
if __name__ == '__main__':
    specific_rows = [201,202,203,204,205]
    Mode = "simple"
    main(specific_rows, Mode)

