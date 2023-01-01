# Number of Female and Male Athletes over time
maleParticipants = olympic_df.loc[(olympic_df['Sex'] == 'M')]
f = femaleParticipants.groupby('Year')['Sex'].value_counts()
m = maleParticipants.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(16, 14))
plt.plot(f.loc[:,'F'], label = 'Female', color = 'red')
plt.plot(m.loc[:,'M'], label = 'Male', color = 'blue')
plt.title('Number of Female and Male Athletes over time', fontweight='bold', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.xticks(np.arange(1890, 2030, step=4))
plt.xticks(rotation = 45)
plt.legend()
plt.show()