mkdir -p ./record
for source in DE007 DE014 DE021 FE007 FE014 FE021 DE FE
do
	for target in DE007 DE014 DE021 FE007 FE014 FE021 DE FE
	do
		rm -r ./utils/__pycache__
		rm -r ./model/__pycache__
		task="$source-$target"
		# echo $task
		python main.py --source $source --target $target --task $task
		sleep 5
	done
done
mkdir ./record/FRAN
mv ./record/*.txt ./record/FRAN/
