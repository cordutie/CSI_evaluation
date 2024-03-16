import preamble.aux as a

originals_path = input("Enter path of original:")
versions_path  = input("Enter path of versions:")

results = a.tester(originals_path, versions_path)

for key, dict_results in results.items():
    a.dict_results_to_plot(dict_results, key)

