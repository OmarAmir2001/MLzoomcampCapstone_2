import json

# Load the existing notebook
with open('notebooks/Eda_and_cleaning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cells to add for EDA with plots
new_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "id": "eda_header",
        "metadata": {},
        "source": [
            "# Exploratory Data Analysis with Visualizations"
        ]
    },
    # Set plotting style
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "plot_style",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Set plotting style\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "sns.set_palette('husl')\n",
            "%matplotlib inline"
        ]
    },
    # Dataset overview
    {
        "cell_type": "markdown",
        "id": "overview_header",
        "metadata": {},
        "source": [
            "## 1. Dataset Overview"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "dataset_shape",
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "print(f\"Total Records: {df.shape[0]:,}\")\n",
            "print(f\"Total Features: {df.shape[1]}\")\n",
            "print(f\"\\nColumn Names: {list(df.columns)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "missing_values",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check for missing values\n",
            "print(\"Missing Values:\")\n",
            "print(df.isnull().sum())\n",
            "print(f\"\\nTotal missing values: {df.isnull().sum().sum()}\")"
        ]
    },
    # Customer Age Analysis
    {
        "cell_type": "markdown",
        "id": "age_header",
        "metadata": {},
        "source": [
            "## 2. Customer Age Distribution"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "age_stats",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Age statistics\n",
            "print(\"Customer Age Statistics:\")\n",
            "print(df['customer_age'].describe())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "age_dist_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Age distribution plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
            "\n",
            "# Histogram\n",
            "axes[0].hist(df['customer_age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')\n",
            "axes[0].set_xlabel('Customer Age')\n",
            "axes[0].set_ylabel('Frequency')\n",
            "axes[0].set_title('Distribution of Customer Age')\n",
            "axes[0].axvline(df['customer_age'].mean(), color='red', linestyle='--', label=f'Mean: {df[\"customer_age\"].mean():.1f}')\n",
            "axes[0].axvline(df['customer_age'].median(), color='green', linestyle='--', label=f'Median: {df[\"customer_age\"].median():.1f}')\n",
            "axes[0].legend()\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Box plot\n",
            "axes[1].boxplot(df['customer_age'], vert=True)\n",
            "axes[1].set_ylabel('Customer Age')\n",
            "axes[1].set_title('Box Plot of Customer Age')\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Customer Gender Analysis
    {
        "cell_type": "markdown",
        "id": "gender_header",
        "metadata": {},
        "source": [
            "## 3. Customer Gender Distribution"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "gender_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Gender distribution\n",
            "print(\"Customer Gender Distribution:\")\n",
            "print(df['customer_gender'].value_counts())\n",
            "print(f\"\\nPercentages:\")\n",
            "print(df['customer_gender'].value_counts(normalize=True) * 100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "gender_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Gender visualization\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Bar plot\n",
            "gender_counts = df['customer_gender'].value_counts()\n",
            "axes[0].bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])\n",
            "axes[0].set_xlabel('Gender')\n",
            "axes[0].set_ylabel('Count')\n",
            "axes[0].set_title('Customer Gender Distribution')\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Pie chart\n",
            "axes[1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])\n",
            "axes[1].set_title('Customer Gender Proportion')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Product Analysis
    {
        "cell_type": "markdown",
        "id": "product_header",
        "metadata": {},
        "source": [
            "## 4. Product Purchased Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "product_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Product distribution\n",
            "print(f\"Number of unique products: {df['product_purchased'].nunique()}\")\n",
            "print(f\"\\nTop 15 Products:\")\n",
            "print(df['product_purchased'].value_counts().head(15))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "product_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Top 15 products visualization\n",
            "plt.figure(figsize=(12, 8))\n",
            "top_products = df['product_purchased'].value_counts().head(15)\n",
            "plt.barh(range(len(top_products)), top_products.values, color='coral')\n",
            "plt.yticks(range(len(top_products)), top_products.index)\n",
            "plt.xlabel('Number of Tickets')\n",
            "plt.title('Top 15 Products by Ticket Count')\n",
            "plt.grid(True, alpha=0.3, axis='x')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Ticket Type Analysis
    {
        "cell_type": "markdown",
        "id": "ticket_type_header",
        "metadata": {},
        "source": [
            "## 5. Ticket Type Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "ticket_type_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ticket type distribution\n",
            "print(\"Ticket Type Distribution:\")\n",
            "print(df['ticket_type'].value_counts())\n",
            "print(f\"\\nPercentages:\")\n",
            "print(df['ticket_type'].value_counts(normalize=True) * 100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "ticket_type_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ticket type visualization\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Bar plot\n",
            "ticket_type_counts = df['ticket_type'].value_counts()\n",
            "axes[0].bar(ticket_type_counts.index, ticket_type_counts.values, color='steelblue')\n",
            "axes[0].set_xlabel('Ticket Type')\n",
            "axes[0].set_ylabel('Count')\n",
            "axes[0].set_title('Ticket Type Distribution')\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Pie chart\n",
            "axes[1].pie(ticket_type_counts.values, labels=ticket_type_counts.index, autopct='%1.1f%%', startangle=90)\n",
            "axes[1].set_title('Ticket Type Proportion')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Ticket Subject Analysis
    {
        "cell_type": "markdown",
        "id": "ticket_subject_header",
        "metadata": {},
        "source": [
            "## 6. Ticket Subject Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "ticket_subject_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ticket subject distribution\n",
            "print(f\"Number of unique ticket subjects: {df['ticket_subject'].nunique()}\")\n",
            "print(f\"\\nTop 15 Ticket Subjects:\")\n",
            "print(df['ticket_subject'].value_counts().head(15))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "ticket_subject_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Top 15 ticket subjects visualization\n",
            "plt.figure(figsize=(12, 8))\n",
            "top_subjects = df['ticket_subject'].value_counts().head(15)\n",
            "plt.barh(range(len(top_subjects)), top_subjects.values, color='mediumseagreen')\n",
            "plt.yticks(range(len(top_subjects)), top_subjects.index)\n",
            "plt.xlabel('Number of Tickets')\n",
            "plt.title('Top 15 Ticket Subjects')\n",
            "plt.grid(True, alpha=0.3, axis='x')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Ticket Priority Analysis
    {
        "cell_type": "markdown",
        "id": "priority_header",
        "metadata": {},
        "source": [
            "## 7. Ticket Priority Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "priority_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ticket priority distribution\n",
            "print(\"Ticket Priority Distribution:\")\n",
            "print(df['ticket_priority'].value_counts())\n",
            "print(f\"\\nPercentages:\")\n",
            "print(df['ticket_priority'].value_counts(normalize=True) * 100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "priority_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Priority visualization with custom order\n",
            "priority_order = ['critical', 'high', 'medium', 'low']\n",
            "priority_counts = df['ticket_priority'].value_counts().reindex(priority_order)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Bar plot\n",
            "colors = ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C']\n",
            "axes[0].bar(priority_counts.index, priority_counts.values, color=colors)\n",
            "axes[0].set_xlabel('Priority Level')\n",
            "axes[0].set_ylabel('Count')\n",
            "axes[0].set_title('Ticket Priority Distribution')\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Pie chart\n",
            "axes[1].pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)\n",
            "axes[1].set_title('Ticket Priority Proportion')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Ticket Channel Analysis
    {
        "cell_type": "markdown",
        "id": "channel_header",
        "metadata": {},
        "source": [
            "## 8. Ticket Channel Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "channel_counts",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ticket channel distribution\n",
            "print(\"Ticket Channel Distribution:\")\n",
            "print(df['ticket_channel'].value_counts())\n",
            "print(f\"\\nPercentages:\")\n",
            "print(df['ticket_channel'].value_counts(normalize=True) * 100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "channel_plot",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Channel visualization\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Bar plot\n",
            "channel_counts = df['ticket_channel'].value_counts()\n",
            "axes[0].bar(channel_counts.index, channel_counts.values, color='orchid')\n",
            "axes[0].set_xlabel('Channel')\n",
            "axes[0].set_ylabel('Count')\n",
            "axes[0].set_title('Ticket Channel Distribution')\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "axes[0].grid(True, alpha=0.3, axis='y')\n",
            "\n",
            "# Pie chart\n",
            "axes[1].pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%', startangle=90)\n",
            "axes[1].set_title('Ticket Channel Proportion')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Bivariate Analysis
    {
        "cell_type": "markdown",
        "id": "bivariate_header",
        "metadata": {},
        "source": [
            "## 9. Bivariate Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "age_by_gender",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Age distribution by gender\n",
            "plt.figure(figsize=(12, 6))\n",
            "for gender in df['customer_gender'].unique():\n",
            "    subset = df[df['customer_gender'] == gender]['customer_age']\n",
            "    plt.hist(subset, bins=20, alpha=0.5, label=gender, edgecolor='black')\n",
            "\n",
            "plt.xlabel('Customer Age')\n",
            "plt.ylabel('Frequency')\n",
            "plt.title('Age Distribution by Gender')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "priority_by_type",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Priority by ticket type\n",
            "priority_type_crosstab = pd.crosstab(df['ticket_type'], df['ticket_priority'])\n",
            "print(\"Ticket Priority by Type:\")\n",
            "print(priority_type_crosstab)\n",
            "\n",
            "# Stacked bar chart\n",
            "priority_type_crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#D32F2F', '#F57C00', '#FBC02D', '#388E3C'])\n",
            "plt.xlabel('Ticket Type')\n",
            "plt.ylabel('Count')\n",
            "plt.title('Ticket Priority Distribution by Ticket Type')\n",
            "plt.legend(title='Priority', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
            "plt.xticks(rotation=45, ha='right')\n",
            "plt.grid(True, alpha=0.3, axis='y')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "channel_by_priority",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Channel by priority\n",
            "channel_priority_crosstab = pd.crosstab(df['ticket_channel'], df['ticket_priority'])\n",
            "print(\"\\nTicket Channel by Priority:\")\n",
            "print(channel_priority_crosstab)\n",
            "\n",
            "# Grouped bar chart\n",
            "channel_priority_crosstab.plot(kind='bar', figsize=(12, 6), color=['#D32F2F', '#F57C00', '#FBC02D', '#388E3C'])\n",
            "plt.xlabel('Ticket Channel')\n",
            "plt.ylabel('Count')\n",
            "plt.title('Ticket Priority Distribution by Channel')\n",
            "plt.legend(title='Priority', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
            "plt.xticks(rotation=45, ha='right')\n",
            "plt.grid(True, alpha=0.3, axis='y')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "gender_by_type",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Gender by ticket type\n",
            "gender_type_crosstab = pd.crosstab(df['ticket_type'], df['customer_gender'])\n",
            "print(\"\\nCustomer Gender by Ticket Type:\")\n",
            "print(gender_type_crosstab)\n",
            "\n",
            "# Stacked bar chart\n",
            "gender_type_crosstab.plot(kind='bar', stacked=False, figsize=(12, 6))\n",
            "plt.xlabel('Ticket Type')\n",
            "plt.ylabel('Count')\n",
            "plt.title('Customer Gender Distribution by Ticket Type')\n",
            "plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
            "plt.xticks(rotation=45, ha='right')\n",
            "plt.grid(True, alpha=0.3, axis='y')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Summary Statistics
    {
        "cell_type": "markdown",
        "id": "summary_header",
        "metadata": {},
        "source": [
            "## 10. Summary Statistics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "summary_stats",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Comprehensive summary\n",
            "print(\"=\"*80)\n",
            "print(\"COMPREHENSIVE DATA SUMMARY\")\n",
            "print(\"=\"*80)\n",
            "print(f\"\\nDataset Shape: {df.shape}\")\n",
            "print(f\"Total Records: {df.shape[0]:,}\")\n",
            "print(f\"Total Features: {df.shape[1]}\")\n",
            "print(f\"\\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\")\n",
            "print(f\"\\nData Types:\")\n",
            "print(df.dtypes.value_counts())\n",
            "print(f\"\\nMissing Values: {df.isnull().sum().sum()}\")\n",
            "print(f\"\\nDuplicate Rows: {df.duplicated().sum()}\")\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"KEY INSIGHTS:\")\n",
            "print(\"=\"*80)\n",
            "print(f\"1. Most common ticket type: {df['ticket_type'].mode()[0]} ({df['ticket_type'].value_counts().iloc[0]} tickets)\")\n",
            "print(f\"2. Most common priority: {df['ticket_priority'].mode()[0]} ({df['ticket_priority'].value_counts().iloc[0]} tickets)\")\n",
            "print(f\"3. Most common channel: {df['ticket_channel'].mode()[0]} ({df['ticket_channel'].value_counts().iloc[0]} tickets)\")\n",
            "print(f\"4. Average customer age: {df['customer_age'].mean():.1f} years\")\n",
            "print(f\"5. Most common gender: {df['customer_gender'].mode()[0]} ({df['customer_gender'].value_counts().iloc[0]} customers)\")\n",
            "print(f\"6. Most problematic product: {df['product_purchased'].mode()[0]} ({df['product_purchased'].value_counts().iloc[0]} tickets)\")\n",
            "print(\"=\"*80)"
        ]
    }
]

# Add new cells to the notebook
nb['cells'].extend(new_cells)

# Save the updated notebook
with open('notebooks/Eda_and_cleaning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully added {len(new_cells)} new cells to the notebook!")
print(f"Total cells now: {len(nb['cells'])}")
