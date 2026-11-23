---
pretty_name: Table-GPT (Table-tuned GPT for Diverse Table Tasks)
configs:
- config_name: All
  data_files:
  - split: train
    path: train/train_All.jsonl
  - split: test
    path: test/test_All.jsonl
- config_name: ColumnAugmentation
  data_files:
  - split: train
    path: train/train_ColumnAugmentation.jsonl
- config_name: ColumnFinding
  data_files:
  - split: test
    path: test/test_ColumnFinding.jsonl
- config_name: ColumnTypeAnnotation
  data_files:
  - split: test
    path: test/test_ColumnTypeAnnotation.jsonl
- config_name: DataImputation
  data_files:
  - split: train
    path: train/train_DataImputation.jsonl
  - split: test
    path: test/test_DataImputation.jsonl
- config_name: EntityMatching
  data_files:
  - split: train
    path: train/train_EntityMatching.jsonl
  - split: test
    path: test/test_EntityMatching.jsonl
- config_name: ErrorDetection
  data_files:
  - split: train
    path: train/train_ErrorDetection.jsonl
  - split: test
    path: test/test_ErrorDetection.jsonl
- config_name: HeaderValueMatching
  data_files:
  - split: train
    path: train/train_HeaderValueMatching.jsonl
- config_name: ListExtraction
  data_files:
  - split: train
    path: train/train_ListExtraction.jsonl
- config_name: MissingValueIdentification
  data_files:
  - split: test
    path: test/test_MissingValueIdentification.jsonl
- config_name: NL2SQL
  data_files:
  - split: train
    path: train/train_NL2SQL.jsonl
- config_name: Row2RowTransformation
  data_files:
  - split: train
    path: train/train_Row2RowTransformation.jsonl
  - split: test
    path: test/test_Row2RowTransformation.jsonl
- config_name: RowAugmentation
  data_files:
  - split: train
    path: train/train_RowAugmentation.jsonl
- config_name: RowColumnFiltering
  data_files:
  - split: train
    path: train/train_RowColumnFiltering.jsonl
- config_name: RowColumnSorting
  data_files:
  - split: train
    path: train/train_RowColumnSorting.jsonl
- config_name: RowColumnSwapping
  data_files:
  - split: train
    path: train/train_RowColumnSwapping.jsonl
- config_name: SchemaMatching
  data_files:
  - split: train
    path: train/train_SchemaMatching.jsonl
  - split: test
    path: test/test_SchemaMatching.jsonl
- config_name: TableQuestion
  data_files:
  - split: test
    path: test/test_TableQuestion.jsonl
- config_name: TableSummarization
  data_files:
  - split: train
    path: train/train_TableSummarization.jsonl
license: mit
---

# Table-GPT: Table-tuned GPT for Diverse Table Tasks

This repository contains training and test datasets for the SIGMOD'24 paper [Table-GPT: Table-tuned GPT for Diverse Table Tasks](https://arxiv.org/abs/2310.09263). The source code for data generation and task evaluation are available here: https://github.com/microsoft/Table-GPT, which can be used to generate more training data for table-related tasks.

## Task Descriptions
We collect (or synthesize) 18 diverse table-related tasks, which are summarized in the table below. There are 14 training tasks (T5 - T18) and 9 test tasks (T1 - T9). Some of these tasks (T-1 to T-4) are used as unseen hold-out tasks, to evaluate Table-GPT ability to generalize to completely new and unseen tasks. Some of these tasks (T-10 to T-18) are used for training only. 
 
| **Task Name** | **Task Description** | **Task Category** | **Train/Test** |
|---|---|---|---|
| T-1: Missing-Value Identification (MV) | Identify the row and column position of the only missing cell in a given table | Table understanding | Test only |
| T-2: Column Finding (CF) | Identify the column-name of a specific value that appears only once in a given table | Table Understanding | Test only |
| T-3: Table-QA (TQA) | Answer a natural-language question based on the content of a table | Table QA | Test only |
| T-4: Column Type Annotation (CTA) | Find the semantic type of a column from a given list of choices | Table understanding | Test only |
| T-5: Row-to-row Transformormation (R2R) | Transform table data based on input/output examples | Data transformation | Train/Test |
| T-6: Entity Matching (EM) | Match rows from two tables that refer to the same real-world entity | Table matching | Train/Test |
| T-7: Schema Matching (SM) | Match columns from two tables that refer to the same meaning | Table matching | Train/Test |
| T-8: Data Imputation (DI) | Predict the missing values in a cell based on the table context | Data cleaning | Train/Test |
| T-9: Error Detection (ED) | Detect data values in a table that is a likely error from misspelling | Data cleaning | Train/Test |
| T-10: List Extraction (LE) | Extract a structured table from a list that lacks explicit column delimiters | Data transformation | Train only |
| T-11: Header Value Matching (HVM) | Match column-headers with its data values drawn from the same table | Table matching | Train only |
| T-12: Natural-Language to SQL (NS) | Translate a natural-language question on a table into a SQL query | NL-to-SQL | Train only |
| T-13: Table Summarization (TS) | Produce a natural-language summary for the content in a table | Data augmentation | Train only |
| T-14: Column Augmentation (CA) | Augment a table with additional columns compatible with a given table | Data augmentation | Train only |
| T-15: Row Augmentation (RA) | Augment a table with additional rows compatible with a given table | Data augmentation | Train only |
| T-16: Row/Column Swapping (RCSW) | Manipulate a given table by swapping the position of two rows or columns | Table manipulation | Train only |
| T-17: Row/Column Filtering (RCF) | Manipulate a given table by filtering on given rows or columns | Table manipulation | Train only |
| T-18: Row/Column Sorting (RCS) | Manipulate a given table by performing sorting on given rows or columns | Table manipulation | Train only |

 ## Structure
### Repository Structure
The structure of this repository is shown as follows.

```
Table-GPT
├── train                  
│   ├── train_All.jsonl          # the merged training data of all training tasks   
│   ├── train_{task_name}.jsonl  # the training data for a specific training task 
│   └── ...                       
│   
├── test  
│   ├── test_All.jsonl           # the merged test data of all test tasks 
│   ├── test_{task_name}.jsonl   # the test data for a specific test task 
│   └── ...   
│ 
└── train_large
    ├── train_large_All.jsonl     # a larger training set with additional data
    ├── train_large_{task_name}.jsonl  # the additional training data for a specific training task 
    └── ...   
```
### Data Structure
Each line in the `.jsonl` file represents a single example, containing the following key items:

- **task**: The name of the task associated with the example.
- **dataset**: The name of the dataset from which the example originates.
- **prompt**: The input prompt provided to the model for generating a response.
- **completion**: The generated output response corresponding to the given prompt.
- **messages**: A list of messages that combine the prompt and completion, typically used in chat-oriented models.
- **metadata**: A dict for other information about the example. 

## Dataset

### Test Set
The datasets used for test tasks are summarized as follows.

| **Task** | **Dataset** | **Size (zero-shot + few-shot)** |
|---|---|---|
| ColumnFinding | Spreadsheets-CF | 1682 |
| ColumnTypeAnnotation | EfthymiouTest | 1188 |
| ColumnTypeAnnotation | LimayeTest | 348 |
| ColumnTypeAnnotation | SherlockTest | 1940 |
| ColumnTypeAnnotation | T2DTest | 734 |
| DataImputation | Spreadsheets-DI | 2000 |
| EntityMatching | Amazon-Google | 4586 |
| EntityMatching | Beer | 182 |
| EntityMatching | DBLP-ACM | 4946 |
| EntityMatching | DBLP-GoogleScholar | 11484 |
| EntityMatching | Fodors-Zagats | 378 |
| EntityMatching | Walmart-Amazon | 4098 |
| EntityMatching | iTunes-Amazon | 218 |
| ErrorDetection | Spreadsheets-ED-Real | 1740 |
| ErrorDetection | WebTables-ED-Real | 864 |
| MissingValueIdentification | Spreadsheets-MVI-ColumnNoSep | 2000 |
| MissingValueIdentification | Spreadsheets-MVI-ColumnSep | 2000 |
| MissingValueIdentification | Spreadsheets-MVI-RowNoSep | 2000 |
| MissingValueIdentification | Spreadsheets-MVI-RowSep | 2000 |
| Row2RowTransformation | BingQL-Other | 102 |
| Row2RowTransformation | BingQL-Unit | 99 |
| Row2RowTransformation | FF-GR-Trifacta | 134 |
| Row2RowTransformation | Headcase | 90 |
| Row2RowTransformation | Stackoverflow | 145 |
| SchemaMatching | DeepM | 14 |
| TableQuestion | SQATest | 360 |
| TableQuestion | WikiTest | 8688 |

### Training Set
The datasets used for training tasks are summarized as follows. (We temporarily remove some synthesized data that was used in our original paper due to licensing reasons).

| **Task** | **Dataset** | **Size (zero-shot + few-shot)** |
|---|---|---|
| ColumnAugmentation | Synthetic-CA | 558 |
| DataImputation | Synthetic-DI | 886 |
| EntityMatching | 784datasets | 2010 |
| ErrorDetection | Synthetic-ED | 852 |
| HeaderValueMatching | Synthetic-HVM | 1064 |
| ListExtraction | Synthetic-LE | 537 |
| NL2SQL | WikiSQL | 994 |
| Row2RowTransformation | Synthetic-R2R | 951 |
| RowAugmentation | Synthetic-RA | 553 |
| RowColumnFiltering | Synthetic-RCF | 603 |
| RowColumnSorting | Synthetic-RCS | 576 |
| RowColumnSwapping | Synthetic-RCSW | 556 |
| SchemaMatching | Synthetic-SM | 2068 |
| TableSummarization | Synthetic-TS | 1014 |

### Training Set (Large)
We provide a larger training set with additional generated data for each task, which is summarized as follows.

| **Task** | **Dataset** | **Size (zero-shot + few-shot)** |
|---|---|---|
| ColumnAugmentation | Synthetic-CA | 2861 |
| DataImputation | Synthetic-DI | 4326 |
| EntityMatching | 784datasets | 10000 |
| ErrorDetection | Synthetic-ED | 4292 |
| HeaderValueMatching | Synthetic-HVM | 5562 |
| ListExtraction | Synthetic-LE | 2821 |
| NL2SQL | WikiSQL | 5000 |
| Row2RowTransformation | Synthetic-R2R | 5000 |
| RowAugmentation | Synthetic-RA | 2828 |
| RowColumnFiltering | Synthetic-RCF | 2863 |
| RowColumnSorting | Synthetic-RCS | 2849 |
| RowColumnSwapping | Synthetic-RCSW | 2831 |
| SchemaMatching | Synthetic-SM | 10000 |
| TableSummarization | Synthetic-TS | 5000 |