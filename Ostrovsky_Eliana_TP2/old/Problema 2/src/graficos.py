from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatterplots(df):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='age', y='area_m2')
    plt.title('Área vs Edad')
    plt.xlabel('Área (m2)')
    plt.ylabel('Edad (años)')

    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='area_m2', y='price')
    plt.title('Área vs Precio')
    plt.xlabel('Área (m2)')
    plt.ylabel('Precio')

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='age', y='price')
    plt.title('Edad vs Precio')
    plt.xlabel('Edad (años)')
    plt.ylabel('Precio')

    plt.tight_layout()
    plt.show()


def plot_histograms(df):
    features = ['area_m2', 'age', 'price']

    for feature in features:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        sns.histplot(df[df['is_house'] == 1][feature], color='blue', label='is_house = 1', alpha=0.5)
        sns.histplot(df[df['is_house'] == 0][feature], color='red', label='is_house = 0', alpha=0.5)
        plt.title(f'Distribución de {feature} (is_house)')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.legend()

        plt.subplot(1, 3, 2)
        sns.histplot(df[df['has_pool'] == 1][feature], color='green', label='has_pool = 1', alpha=0.5)
        sns.histplot(df[df['has_pool'] == 0][feature], color='orange', label='has_pool = 0', alpha=0.5)
        plt.title(f'Distribución de {feature} (has_pool)')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.legend()

        plt.subplot(1, 3, 3)
        sns.histplot(df[df['area_units'] == 1][feature], color='violet', label='area_units = sqft', alpha=0.5)
        sns.histplot(df[df['area_units'] == 0][feature], color='yellow', label='area_units = m2', alpha=0.5)
        plt.title(f'Distribución de {feature} (area_units)')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.legend()

        plt.tight_layout()
        plt.show()

def plot_lat_lon(df):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='lon', y='lat')
    plt.title('Latitud vs Longitud (Todos los datos)')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df[df['area_units'] == 0], x='lon', y='lat')
    plt.title('Latitud vs Longitud (m2)')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df[df['area_units'] == 1], x='lon', y='lat')
    plt.title('Latitud vs Longitud (sqft)')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    plt.tight_layout()
    plt.show()

def area_vs_rooms(df):
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  

    ax1 = plt.subplot(gs[0])
    sns.scatterplot(data=df, x='area_m2', y='rooms', ax=ax1)
    ax1.set_title('Área vs Número de Habitaciones')
    ax1.set_xlabel('Área (m2)')
    ax1.set_ylabel('Número de Habitaciones')
    ax2 = plt.subplot(gs[1])

    for room in range(1, 6):
        subset = df[df['rooms'] == room]
        sns.histplot(subset['area_m2'], label=f'{room} habitaciones', alpha=0.5, kde=False, ax=ax2, bins=int((subset['area_m2'].max() - subset['area_m2'].min()) // 4))

    ax2.set_title('Distribución del Área por Número de Habitaciones')
    ax2.set_xlabel('Área (m2)')
    ax2.set_ylabel('Frecuencia')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    