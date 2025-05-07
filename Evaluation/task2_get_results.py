import os
import argparse


def calculated_ndcg(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = os.popen(f"{trec_eval_tool} {qre_file_path} {res_directory + file} -m ndcg_cut.10").read()
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_p_at_10(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = os.popen(f"{trec_eval_tool} {qre_file_path} {res_directory + file} -l2 -m P.10").read()
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def calculated_p_at_5(res_directory, trec_eval_tool, qre_file_path):
    result = {}
    for file in os.listdir(res_directory):
        output = os.popen(f"{trec_eval_tool} {qre_file_path} {res_directory + file} -l2 -m P.5").read()
        score = output.split("\t")[2].strip()
        submission = file.split(".")[0].split("prime_")[1]
        result[submission] = score
    return result


def get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file):
    with open(evaluation_result_file, "w") as file_res:
        res_ndcg_10 = calculated_ndcg(prim_result_dir, trec_eval_tool, qre_file_path)
        res_p10 = calculated_p_at_10(prim_result_dir, trec_eval_tool, qre_file_path)
        res_p5 = calculated_p_at_5(prim_result_dir, trec_eval_tool, qre_file_path)
        file_res.write("System\tnDCG'10\tp@5\tp@10\n")
        for sub in res_ndcg_10:
            file_res.write(f"{sub}\t{res_ndcg_10[sub]}\t{res_p5[sub]}\t{res_p10[sub]}\n")


def main():
    """
    Sample command :
     python task2_get_results.py -eva trec_eval -qre qrel_task2_2021.tsv -pri Task2_2021_Prime/ -res 2021_task2.tsv
    """
    parser = argparse.ArgumentParser(description='Specify the trec_eval file path, qrel file, '
                                                 'deduplicate results directory and result file path')

    parser.add_argument('-eva', help='trec_eval tool file path', required=True)
    parser.add_argument('-qre', help='qrel file path', required=True)
    parser.add_argument('-pri', help='prime results directory', required=True)
    parser.add_argument('-res', help='evaluation result file', required=True)
    args = vars(parser.parse_args())
    trec_eval_tool = args['eva']
    qre_file_path = args['qre']
    prim_result_dir = args['pri'] + "/"
    evaluation_result_file = args['res']

    get_result(trec_eval_tool, qre_file_path, prim_result_dir, evaluation_result_file)


if __name__ == "__main__":
    main()
