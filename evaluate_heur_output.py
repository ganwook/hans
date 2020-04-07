import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", default="spinn_preds_heur.txt", type=str, # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_dir", default="heuristics_evaluation_set.txt", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default='/', type=str, # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    def format_label(label):
        if label == "entailment":
            return "entailment"
        else:
            return "non-entailment"
    
    results = {}

    fi = open(args.input_dir, "r")
    first = True
    guess_dict = {}
    for line in fi:
        if first:
            first = False
            continue
        else:
            parts = line.strip().split(",")
            guess_dict[parts[0]] = format_label(parts[1])

    fi = open(args.eval_dir, "r")
    correct_dict = {}
    first = True
    heuristic_list = []
    subcase_list = []
    template_list = []

    for line in fi:
        if first:
            labels = line.strip().split("\t")
            idIndex = labels.index("pairID")
            first = False
            continue
        else:
            parts = line.strip().split("\t")
            this_line_dict = {}
            for index, label in enumerate(labels):
                if label == "pairID":
                    continue
                else:
                    this_line_dict[label] = parts[index]
            correct_dict[parts[idIndex]] = this_line_dict
            
            if this_line_dict["heuristic"] not in heuristic_list:
                heuristic_list.append(this_line_dict["heuristic"])
            if this_line_dict["subcase"] not in subcase_list:
                subcase_list.append(this_line_dict["subcase"])
            if this_line_dict["template"] not in template_list:
                template_list.append(this_line_dict["template"])

    heuristic_ent_correct_count_dict = {}
    subcase_correct_count_dict = {}
    template_correct_count_dict = {}
    heuristic_ent_incorrect_count_dict = {}
    subcase_incorrect_count_dict = {}
    template_incorrect_count_dict = {}
    heuristic_nonent_correct_count_dict = {}
    heuristic_nonent_incorrect_count_dict = {}

    for heuristic in heuristic_list:
        heuristic_ent_correct_count_dict[heuristic] = 0
        heuristic_ent_incorrect_count_dict[heuristic] = 0
        heuristic_nonent_correct_count_dict[heuristic] = 0 
        heuristic_nonent_incorrect_count_dict[heuristic] = 0

    for subcase in subcase_list:
        subcase_correct_count_dict[subcase] = 0
        subcase_incorrect_count_dict[subcase] = 0

    for template in template_list:
        template_correct_count_dict[template] = 0
        template_incorrect_count_dict[template] = 0

    for key in correct_dict:
        traits = correct_dict[key]
        heur = traits["heuristic"]
        subcase = traits["subcase"]
        template = traits["template"]
        
        guess = guess_dict[key]
        correct = traits["gold_label"]

        if guess == correct:
            if correct == "entailment":
                heuristic_ent_correct_count_dict[heur] += 1
            else:
                heuristic_nonent_correct_count_dict[heur] += 1
                
            subcase_correct_count_dict[subcase] += 1
            template_correct_count_dict[template] += 1
        else:
            if correct == "entailment":
                heuristic_ent_incorrect_count_dict[heur] += 1
            else:
                heuristic_nonent_incorrect_count_dict[heur] += 1
            subcase_incorrect_count_dict[subcase] += 1
            template_incorrect_count_dict[template] += 1

    ent_correct = 0
    ent_incorrect = 0

    print("Heuristic entailed results:")
    for heuristic in heuristic_list:
        correct = heuristic_ent_correct_count_dict[heuristic]
        incorrect = heuristic_ent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        percent = correct * 1.0 / total
        print(heuristic + ": " + str(percent))
        ent_correct += correct
        ent_incorrect += incorrect

    acc_ent = ent_correct * 1.0/ (ent_correct + ent_incorrect)
    results['acc_ent'] = acc_ent
    print(f'acc-ent : {str(acc_ent)}')
    
    not_correct = 0
    not_incorrect = 0

    print("")
    print("Heuristic non-entailed results:")
    for heuristic in heuristic_list:
        correct = heuristic_nonent_correct_count_dict[heuristic]
        incorrect = heuristic_nonent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        percent = correct * 1.0 / total
        print(heuristic + ": " + str(percent), "| total : ", total)
        not_correct += correct
        not_incorrect += incorrect
    
    acc_not = not_correct * 1.0 / (not_correct + not_incorrect)
    results['acc_not'] = acc_not
    print(f'acc-not : {str(acc_not)}')

    acc_total = (ent_correct + not_correct) * 1.0 / (not_correct + not_incorrect + ent_correct + ent_incorrect)
    results['acc_total'] = acc_total
    print(f'acc-total : {str(acc_total)}')

    print("")
    print("Subcase results:")
    for subcase in subcase_list:
        correct = subcase_correct_count_dict[subcase]
        incorrect = subcase_incorrect_count_dict[subcase]
        total = correct + incorrect
        percent = correct * 1.0 / total
        print(subcase + ": " + str(percent))
        results[subcase] = percent

    ## save the result
    with open(args.output_dir, 'w') as f:
        for k, v in results.items():
            f.write(f'{k},{v}\n')

if __name__ == '__main__':
    main()