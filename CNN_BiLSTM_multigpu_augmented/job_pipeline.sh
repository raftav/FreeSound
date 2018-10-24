#!/bin/bash

for exp in 2; do
	id1=`sbatch --parsable run_model.slurm $exp`
	echo "started exp $exp with id $id1"
	id2=`sbatch --parsable --dependency=afterany:$id1 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id2. To start after job $id1 ends"
	id3=`sbatch --parsable --dependency=afterany:$id2 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id3. To start after job $id2 ends"
	id4=`sbatch --parsable --dependency=afterany:$id3 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id4. To start after job $id3 ends"
	id5=`sbatch --parsable --dependency=afterany:$id4 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id5. To start after job $id4 ends"
	id6=`sbatch --parsable --dependency=afterany:$id5 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id6. To start after job $id5 ends"
	id7=`sbatch --parsable --dependency=afterany:$id6 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id7. To start after job $id6 ends"
	id8=`sbatch --parsable --dependency=afterany:$id7 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id8. To start after job $id7 ends"
	id9=`sbatch --parsable --dependency=afterany:$id8 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id9. To start after job $id8 ends"
	id10=`sbatch --parsable --dependency=afterany:$id9 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id10. To start after job $id9 ends"
	id11=`sbatch --parsable --dependency=afterany:$id10 restore_model.slurm $exp`
	echo "submitted exp $exp with id $id11. To start after job $id10 ends"

	#id12=`sbatch --parsable --dependency=afterany:$id11 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id12. To start after job $id11 ends"
	#id13=`sbatch --parsable --dependency=afterany:$id12 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id13. To start after job $id12 ends"
	#id14=`sbatch --parsable --dependency=afterany:$id13 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id14. To start after job $id13 ends"
	#id15=`sbatch --parsable --dependency=afterany:$id14 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id15. To start after job $id14 ends"
	#id16=`sbatch --parsable --dependency=afterany:$id15 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id16. To start after job $id15 ends"
	#id17=`sbatch --parsable --dependency=afterany:$id16 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id17. To start after job $id16 ends"
	#id8=`sbatch --parsable --dependency=afterany:$id17 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id18. To start after job $id17 ends"
	#id19=`sbatch --parsable --dependency=afterany:$id18 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id19. To start after job $id18 ends"
	#id20=`sbatch --parsable --dependency=afterany:$id19 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id20. To start after job $id19 ends"
	#id21=`sbatch --parsable --dependency=afterany:$id20 restore_model.slurm $exp`
	#echo "submitted exp $exp with id $id21. To start after job $id20 ends"
done
