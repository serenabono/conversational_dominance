import os
import pandas as pd
import re

# Get the current file path
current_file_path = os.path.abspath(__file__)
folder_path = os.path.dirname(os.path.dirname(current_file_path))


def get_transcripts_df():
    # Get a list of all files in the folder
    dir_path = os.path.join(folder_path, 'data/external/transcriptions')
    all_files = os.listdir(dir_path)
    # Filter files ending with .txt
    txt_files = [file for file in all_files if file.endswith('.txt')]

    file_name_list = []
    file_content_list = []

    for file_name in txt_files:

        file_path = os.path.join(dir_path, file_name)

        # Open the file in read mode ('r')
        with open(file_path, 'r') as file:
        # Read all lines, remove newline characters, and concatenate into a single string
            file_content = ' '.join(line.strip() for line in file)

        file_name_list.append(file_name.split('.')[0])
        file_content_list.append(file_content)

    data = {'file_name': file_name_list, 'file_content': file_content_list}
    df = pd.DataFrame(data)

    return df


def get_dominance_scores_df():

    #dominance_assessment_file_path = '/Users/nishthasardana/Downloads/Dominance-assesment.xlsx'
    dominance_assessment_file_path = os.path.join(folder_path, 'data/external/Dominance-assesment.xlsx')
    dominance_df = pd.read_excel(dominance_assessment_file_path, skiprows=3, sheet_name='raw')
    dominance_df.drop(dominance_df.columns[6], axis=1, inplace=True)
    # Generate column names
    columns = ['file_name']
    num_speakers = 2
    num_columns_per_speaker = 5

    for i in range(1, num_speakers + 1):
        for j in range(1, num_columns_per_speaker + 1):
            column_name = f'speaker_{i}_{j}'
            columns.append(column_name)

    # Assign the new column names to the DataFrame
    dominance_df.columns = columns
    dominance_df.dropna(inplace=True)

    dominance_df['speaker_1_dom_score'] = dominance_df.loc[:, dominance_df.columns.str.startswith('speaker_1')].mean(axis=1)
    dominance_df['speaker_2_dom_score'] = dominance_df.loc[:, dominance_df.columns.str.startswith('speaker_2')].mean(axis=1)

    return dominance_df

def get_particpants_df():
    participant_file_path = os.path.join(folder_path, 'data/external/participants.xlsx')
    df = pd.read_excel(participant_file_path, sheet_name='Sheet1')
    df = df.drop(df.columns[-1], axis=1)
    df.columns = ['file_name', 'participant_id']
    df.sort_values(['file_name', 'participant_id'], inplace=True)
    df['participant'] = 'speaker_' + df.groupby('file_name').cumcount().add(1).astype(str)
    df = df.pivot(index='file_name', columns='participant', values='participant_id').reset_index()
    return df


def main():
    # Get the current file path
    current_file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(os.path.dirname(current_file_path))
    file_save_path = os.path.join(folder_path, 'data/processed/transcript_dominance.csv')
    
    df_dominance = get_transcripts_df()
    df_transcripts = get_dominance_scores_df()
    df_participants = get_particpants_df()
    df = df_transcripts.merge(df_dominance, on='file_name')
    result_df = df.merge(df_participants, on='file_name')

    def remove_words_with_hashtags_and_brackets(input_string):
        regex  = r'[\w\.\+]*\[[\w\.\+]*\][\w\.\+]*|[\w\.\+]*#[\w\.\+]*'
        result = re.sub(regex, '', input_string)
        return result

    def replace_pattern_with_mod(input_string):
        regex = r'M0[0-9][0-9]_S[0-9][0-9]'
        result = re.sub(regex, '[MOD]', input_string)
        return result

    def replace_person_with_p1(row):
        return row['file_content'].replace(row['speaker_1'], '[SPK1]')

    def replace_person_with_p2(row):
        return row['file_content'].replace(row['speaker_2'], '[SPK2]')
    
    result_df['file_content'] = result_df['file_content'].apply(remove_words_with_hashtags_and_brackets)

    result_df['file_content'] = result_df['file_content'].apply(replace_pattern_with_mod)

    result_df['file_content'] = result_df.apply(replace_person_with_p1, axis=1)
    result_df['file_content'] = result_df.apply(replace_person_with_p2, axis=1)

    result_df.to_csv(file_save_path, index=False)


if __name__ == "__main__":
    main()





