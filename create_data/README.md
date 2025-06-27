# Create your own test dataset (CPU)

If you want to use this model to analyze your own experimental materials, you must convert your data into the following specific formats:
(All raw data should be placed in the ./data/original_data directory.)

## Convert XRD patterns

Regardless of the original format of your XRD files, convert them into a _.pt_ file that contains a single line of relative intensity values with a step size of 0.02 (2θ):
1. The file should contain only one column, representing relative intensity.
2. The data should have exactly 5250 rows, corresponding to 2θ values ranging from 5° to 110°, with a step size of 0.02°.

## Convert metal nodes and organic linkers

Prepare your prior information about metal nodes and organic linkers in a .csv file with the following requirements:
1. The file must contain the following six columns:"Materials_name",  "Metal_nodes", "Organic_linkers", "Linker_proportion", "Metal_valence", and "Linker_valence" ("Linker_valence" can be left blank; if there is only one type of linker, "Linker_proportion" can also be left blank.)
2. The "Materials_name" column should provide a unique identifier for each structure.
3. The "Metal_nodes" column should list the metal node elements for the structure, separated by commas.
4. The "Organic_linkers" column should provide the SMILES strings of the organic linkers, separated by commas.

Replace ${raw_path} with the actual path to your project directory.
```
python SD-MOFNet-main/create_data/metal_nodes.py --input_csv ${raw_path}/SD-MOFNet-main/data/original_data/original_data.csv --output_csv ${raw_path}/SD-MOFNet-main/data/original_data
python SD-MOFNet-main/create_data/organic_linkers.py --input_csv ${raw_path}/SD-MOFNet-main/data/original_data/original_data.csv --output_dir ${raw_path}/SD-MOFNet-main/data/original_data
python SD-MOFNet-main/create_data/create_dataset.py --input_csv  ${raw_path}/SD-MOFNet-main/data/original_data/original_data.csv --output_pt ${raw_path}/SD-MOFNet-main/data/dataset.pt --error_log_file ${raw_path}/SD-MOFNet-main/data/original_data/log.txt
```
