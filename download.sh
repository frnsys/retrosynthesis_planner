cd data
wget https://ndownloader.figshare.com/articles/5104873/versions/1 -O uspto_reactions.zip
unzip uspto_reactions.zip
7z x 1976_Sep2016_USPTOgrants_smiles.7z
7z x 2001_Sep2016_USPTOapplications_smiles.7z
rm *.{7z,zip}
tail -n +2 1976_Sep2016_USPTOgrants_smiles.rsmi | cut -f1 >> reactions.rsmi
tail -n +2 2001_Sep2016_USPTOapplications_smiles.rsmi | cut -f1 >> reactions.rsmi
echo $(cat reactions.rsmi | wc -l) reactions
