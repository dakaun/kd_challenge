# Main file for running the analysis

import IOHandler as io

def main():
    print("do something")
    df = io.read_data()
    df.head()

main()