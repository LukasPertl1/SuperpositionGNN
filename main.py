from ExperimentalPipeline1 import main

# Specific rows correspond to configurations defined in the CSV files (See ExperimentList folder).
if __name__ == '__main__':
    specific_rows = [223,224,225]
    Mode = "simple"
    main(specific_rows, Mode)

