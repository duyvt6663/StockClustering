import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import scipy.stats as st
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform
import networkx as nx
import riskfolio.src.RiskFunctions as rk
import riskfolio.src.AuxFunctions as af
import riskfolio.src.DBHT as db
import riskfolio.src.GerberStatistic as gs
import DBHTss as dbs

def plot_dendro(
    returns,
    custom_cov=None,
    codependence="pearson",
    linkage="ward",
    k=None,
    max_k=10,
    bins_info="KN",
    alpha_tail=0.05,
    gs_threshold=0.5,
    leaf_order=True,
    show_clusters=True,
    title="",
    height=5,
    width=12,
    ax=None,
    method=None
):
    r"""
    Create a dendrogram based on the selected codependence measure.

    Parameters
    ----------
    returns : DataFrame
        Assets returns.
    custom_cov : DataFrame or None, optional
        Custom covariance matrix, used when codependence parameter has value
        'custom_cov'. The default is None.
    codependence : str, can be {'pearson', 'spearman', 'abs_pearson', 'abs_spearman', 'distance', 'mutual_info', 'tail' or 'custom_cov'}
        The codependence or similarity matrix used to build the distance
        metric and clusters. The default is 'pearson'. Possible values are:

        - 'pearson': pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho_{i,j})}`.
        - 'spearman': spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho_{i,j})}`.
        - 'kendall': kendall correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{kendall}_{i,j})}`.
        - 'gerber1': Gerber statistic 1 correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{gerber1}_{i,j})}`.
        - 'gerber2': Gerber statistic 2 correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{gerber2}_{i,j})}`.
        - 'abs_pearson': absolute value pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'abs_spearman': absolute value spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'abs_kendall': absolute value kendall correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho^{kendall}_{i,j}|)}`.
        - 'distance': distance correlation matrix. Distance formula :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'mutual_info': mutual information matrix. Distance used is variation information matrix.
        - 'tail': lower tail dependence index matrix. Dissimilarity formula :math:`D_{i,j} = -\log{\lambda_{i,j}}`.
        - 'custom_cov': use custom correlation matrix based on the custom_cov parameter. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{pearson}_{i,j})}`.

    linkage : string, optional
        Linkage method of hierarchical clustering, see `linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage>`_ for more details.
        The default is 'ward'. Possible values are:

        - 'single'.
        - 'complete'.
        - 'average'.
        - 'weighted'.
        - 'centroid'.
        - 'median'.
        - 'ward'.
        - 'DBHT': Direct Bubble Hierarchical Tree.

    k : int, optional
        Number of clusters. This value is took instead of the optimal number
        of clusters calculated with the two difference gap statistic.
        The default is None.
    max_k : int, optional
        Max number of clusters used by the two difference gap statistic
        to find the optimal number of clusters. The default is 10.
    bins_info: int or str
        Number of bins used to calculate variation of information. The default
        value is 'KN'. Possible values are:

        - 'KN': Knuth's choice method. See more in `knuth_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html>`_.
        - 'FD': Freedman–Diaconis' choice method. See more in `freedman_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.freedman_bin_width.html>`_.
        - 'SC': Scotts' choice method. See more in `scott_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.scott_bin_width.html>`_.
        - 'HGR': Hacine-Gharbi and Ravier' choice method.
        - int: integer value choice by user.

    alpha_tail : float, optional
        Significance level for lower tail dependence index. The default is 0.05.
    gs_threshold : float, optional
        Gerber statistic threshold. The default is 0.5.
    leaf_order : bool, optional
        Indicates if the cluster are ordered so that the distance between
        successive leaves is minimal. The default is True.
    show_clusters : bool, optional
        Indicates if clusters are plot. The default is True.
    title : str, optional
        Title of the chart. The default is "".
    height : float, optional
        Height of the image in inches. The default is 5.
    width : float, optional
        Width of the image in inches. The default is 12.
    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib axis
        Returns the Axes object with the plot for further tweaking.

    Rdict: A dictionary of data structures computed to render the dendrogram. 
          Its has the following keys:

          'color_list': A list of color names. 
                  The k’th element represents the color of the k’th link.

          'icoord' and 'dcoord': Each of them is a list of lists. 
                  Let icoord = [I1, I2, ..., Ip] where Ik = [xk1, xk2, xk3, xk4] 
                  and dcoord = [D1, D2, ..., Dp] where Dk = [yk1, yk2, yk3, yk4], 
                  then the k’th link painted is (xk1, yk1) - (xk2, yk2) - (xk3, yk3) - (xk4, yk4).

          'ivl': A list of labels corresponding to the leaf nodes.

          'leaves': For each i, H[i] == j, cluster node j appears in position i 
                  in the left-to-right traversal of the leaves, where j < 2n-1
                  and i < n. If j is less than n, the i-th leaf node corresponds 
                  to an original observation. Otherwise, it corresponds to a non-singleton cluster.

          'leaves_color_list': A list of color names. The k’th element represents 
                  the color of the k’th leaf.
    Example
    -------
    ::

        ax = rp.plot_dendrogram(returns=Y, codependence='spearman',
                                linkage='ward', k=None, max_k=10,
                                leaf_order=True, ax=None)

    .. image:: images/Assets_Dendrogram.png


    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()

    labels = np.array(returns.columns.tolist())

    # Calculating codependence matrix and distance metric
    if codependence in {"pearson", "spearman", "kendall"}:
        codep = returns.corr(method=codependence)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
    elif codependence == "gerber1":
        codep = gs.gerber_cov_stat1(returns, threshold=gs_threshold)
        codep = af.cov2corr(codep)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
    elif codependence == "gerber2":
        codep = gs.gerber_cov_stat2(returns, threshold=gs_threshold)
        codep = af.cov2corr(codep)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
    elif codependence in {"abs_pearson", "abs_spearman", "abs_kendall"}:
        codep = np.abs(returns.corr(method=codependence[4:]))
        dist = np.sqrt(np.clip((1 - codep), a_min=0.0, a_max=1.0))
    elif codependence in {"distance"}:
        codep = af.dcorr_matrix(returns).astype(float)
        dist = np.sqrt(np.clip((1 - codep), a_min=0.0, a_max=1.0))
    elif codependence in {"mutual_info"}:
        codep = af.mutual_info_matrix(returns, bins_info).astype(float)
        dist = af.var_info_matrix(returns, bins_info).astype(float)
    elif codependence in {"tail"}:
        codep = af.ltdi_matrix(returns, alpha_tail).astype(float)
        dist = -np.log(codep)
    elif codependence in {"custom_cov"}:
        codep = af.cov2corr(custom_cov).astype(float)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))

    # Hierarchical clustering
    dist = dist.to_numpy()
    dist = pd.DataFrame(dist, columns=codep.columns, index=codep.index)
    if linkage == "DBHT":
        # different choices for D, S give different outputs!
        D = dist.to_numpy()  # dissimilarity matrix
        if codependence in {"pearson", "spearman", "custom_cov"}:
            S = (1 - dist**2).to_numpy()
        else:
            S = codep.copy().to_numpy()  # similarity matrix
        (_, _, _, _, _, clustering) = dbs.DBHTs(
            D, S, method, leaf_order=leaf_order
        )  # DBHT clustering
    else:
        p_dist = squareform(dist, checks=False)
        clustering = hr.linkage(p_dist, method=linkage, optimal_ordering=leaf_order)

    # Ordering clusterings
    permutation = hr.leaves_list(clustering)
    permutation = permutation.tolist()

    if show_clusters is False:
        color_threshold = 0
    elif show_clusters is True:
        # optimal number of clusters
        if k is None:
            k = af.two_diff_gap_stat(codep, dist, clustering, max_k)

        root, nodes = hr.to_tree(clustering, rd=True)
        nodes = [i.dist for i in nodes]
        nodes.sort()
        nodes = nodes[::-1][: k - 1]
        color_threshold = np.min(nodes)
        colors = af.color_list(k)  # color list
        hr.set_link_color_palette(colors)

    R = hr.dendrogram(
        clustering, color_threshold=color_threshold, above_threshold_color="grey", ax=ax
    ) # refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    hr.set_link_color_palette(None)

    ax.set_xticklabels(labels[permutation], rotation=90, ha="center")

    if show_clusters is True:
        i = 0
        for coll in ax.collections[:-1]:  # the last collection is the ungrouped level
            xmin, xmax = np.inf, -np.inf
            ymax = -np.inf
            for p in coll.get_paths():
                (x0, _), (x1, y1) = p.get_extents().get_points()
                xmin = min(xmin, x0)
                xmax = max(xmax, x1)
                ymax = max(ymax, y1)
            rec = plt.Rectangle(
                (xmin - 4, 0),
                xmax - xmin + 8,
                ymax * 1.05,
                facecolor=colors[i],  # coll.get_color()[0],
                alpha=0.2,
                edgecolor="none",
            )
            ax.add_patch(rec)
            i += 1

    ax.set_yticks([])
    ax.set_yticklabels([])
    for i in {"right", "left", "top", "bottom"}:
        side = ax.spines[i]
        side.set_visible(False)

    if title == "":
        title = (
            "Assets Dendrogram ("
            + codependence.capitalize()
            + " & "
            + linkage
            + " " + method 
            + " linkage)"
        )

    ax.set_title(title)

    try:
        fig.tight_layout()
    except:
        pass

    return ax, R

def plot_clus(
    returns,
    custom_cov=None,
    codependence="pearson",
    linkage="ward",
    k=None,
    max_k=10,
    bins_info="KN",
    alpha_tail=0.05,
    gs_threshold=0.5,
    leaf_order=True,
    show_clusters=True,
    dendrogram=True,
    cmap="RdYlBu",
    linecolor="fuchsia",
    title="",
    height=12,
    width=12,
    ax=None,
    method=None
):
    r"""
    Create a clustermap plot based on the selected codependence measure.

    Parameters
    ----------
    returns : DataFrame
        Assets returns.
    custom_cov : DataFrame or None, optional
        Custom covariance matrix, used when codependence parameter has value
        'custom_cov'. The default is None.
    codependence : str, can be {'pearson', 'spearman', 'abs_pearson', 'abs_spearman', 'distance', 'mutual_info', 'tail' or 'custom_cov'}
        The codependence or similarity matrix used to build the distance
        metric and clusters. The default is 'pearson'. Possible values are:

        - 'pearson': pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho_{i,j})}`.
        - 'spearman': spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho_{i,j})}`.
        - 'kendall': kendall correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{kendall}_{i,j})}`.
        - 'gerber1': Gerber statistic 1 correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{gerber1}_{i,j})}`.
        - 'gerber2': Gerber statistic 2 correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{gerber2}_{i,j})}`.
        - 'abs_pearson': absolute value pearson correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'abs_spearman': absolute value spearman correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'abs_kendall': absolute value kendall correlation matrix. Distance formula: :math:`D_{i,j} = \sqrt{(1-|\rho^{kendall}_{i,j}|)}`.
        - 'distance': distance correlation matrix. Distance formula :math:`D_{i,j} = \sqrt{(1-|\rho_{i,j}|)}`.
        - 'mutual_info': mutual information matrix. Distance used is variation information matrix.
        - 'tail': lower tail dependence index matrix. Dissimilarity formula :math:`D_{i,j} = -\log{\lambda_{i,j}}`.
        - 'custom_cov': use custom correlation matrix based on the custom_cov parameter. Distance formula: :math:`D_{i,j} = \sqrt{0.5(1-\rho^{pearson}_{i,j})}`.

    linkage : string, optional
        Linkage method of hierarchical clustering, see `linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html?highlight=linkage#scipy.cluster.hierarchy.linkage>`_ for more details.
        The default is 'ward'. Possible values are:

        - 'single'.
        - 'complete'.
        - 'average'.
        - 'weighted'.
        - 'centroid'.
        - 'median'.
        - 'ward'.
        - 'DBHT': Direct Bubble Hierarchical Tree.

    k : int, optional
        Number of clusters. This value is took instead of the optimal number
        of clusters calculated with the two difference gap statistic.
        The default is None.
    max_k : int, optional
        Max number of clusters used by the two difference gap statistic
        to find the optimal number of clusters. The default is 10.
    bins_info: int or str
        Number of bins used to calculate variation of information. The default
        value is 'KN'. Possible values are:

        - 'KN': Knuth's choice method. See more in `knuth_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html>`_.
        - 'FD': Freedman–Diaconis' choice method. See more in `freedman_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.freedman_bin_width.html>`_.
        - 'SC': Scotts' choice method. See more in `scott_bin_width <https://docs.astropy.org/en/stable/api/astropy.stats.scott_bin_width.html>`_.
        - 'HGR': Hacine-Gharbi and Ravier' choice method.
        - int: integer value choice by user.

    alpha_tail : float, optional
        Significance level for lower tail dependence index. The default is 0.05.
    gs_threshold : float, optional
        Gerber statistic threshold. The default is 0.5.
    leaf_order : bool, optional
        Indicates if the cluster are ordered so that the distance between
        successive leaves is minimal. The default is True.
    show_clusters : bool, optional
        Indicates if clusters are plot. The default is True.
    dendrogram : bool, optional
        Indicates if the plot has or not a dendrogram. The default is True.
    cmap : str or cmap, optional
        Colormap used to plot the pcolormesh plot. The default is 'viridis'.
    linecolor : str, optional
        Color used to identify the clusters in the pcolormesh plot.
        The default is fuchsia'.
    title : str, optional
        Title of the chart. The default is "".
    height : float, optional
        Height of the image in inches. The default is 12.
    width : float, optional
        Width of the image in inches. The default is 12.
    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib axis
        Returns the Axes object with the plot for further tweaking.

    Example
    -------
    ::

        ax = rp.plot_clusters(returns=Y, codependence='spearman',
                              linkage='ward', k=None, max_k=10,
                              leaf_order=True, dendrogram=True, ax=None)

    .. image:: images/Assets_Clusters.png


    """

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if ax is None:
        fig = plt.gcf()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()
        ax.grid(False)
        ax.axis("off")

    labels = np.array(returns.columns.tolist())

    vmin, vmax = 0, 1
    # Calculating codependence matrix and distance metric
    if codependence in {"pearson", "spearman", "kendall"}:
        codep = returns.corr(method=codependence)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
        vmin, vmax = -1, 1
    elif codependence == "gerber1":
        codep = gs.gerber_cov_stat1(returns, threshold=gs_threshold)
        codep = af.cov2corr(codep)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
        vmin, vmax = -1, 1
    elif codependence == "gerber2":
        codep = gs.gerber_cov_stat2(returns, threshold=gs_threshold)
        codep = af.cov2corr(codep)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))
        vmin, vmax = -1, 1
    elif codependence in {"abs_pearson", "abs_spearman", "abs_kendall"}:
        codep = np.abs(returns.corr(method=codependence[4:]))
        dist = np.sqrt(np.clip((1 - codep), a_min=0.0, a_max=1.0))
    elif codependence in {"distance"}:
        codep = af.dcorr_matrix(returns).astype(float)
        dist = np.sqrt(np.clip((1 - codep), a_min=0.0, a_max=1.0))
    elif codependence in {"mutual_info"}:
        codep = af.mutual_info_matrix(returns, bins_info).astype(float)
        dist = af.var_info_matrix(returns, bins_info).astype(float)
    elif codependence in {"tail"}:
        codep = af.ltdi_matrix(returns, alpha_tail).astype(float)
        dist = -np.log(codep)
    elif codependence in {"custom_cov"}:
        codep = af.cov2corr(custom_cov).astype(float)
        dist = np.sqrt(np.clip((1 - codep) / 2, a_min=0.0, a_max=1.0))

    # Hierarchical clustering
    dist = dist.to_numpy()
    dist = pd.DataFrame(dist, columns=codep.columns, index=codep.index)
    dim = len(dist)
    if linkage == "DBHT":
        # different choices for D, S give different outputs!
        D = dist.to_numpy()  # dissimilarity matrix
        if codependence in {"pearson", "spearman", "custom_cov"}:
            S = (1 - dist**2).to_numpy()
        else:
            S = codep.to_numpy()  # similarity matrix
        (_, _, _, _, _, clustering) = dbs.DBHTs(
            D, S, method, leaf_order=leaf_order
        )  # DBHT clustering
    else:
        p_dist = squareform(dist, checks=False)
        clustering = hr.linkage(p_dist, method=linkage, optimal_ordering=leaf_order)

    # Ordering clusterings
    permutation = hr.leaves_list(clustering)
    permutation = permutation.tolist()
    ordered_codep = codep.to_numpy()[permutation, :][:, permutation]

    # optimal number of clusters
    if k is None:
        k = af.two_diff_gap_stat(codep, dist, clustering, max_k)

    clustering_inds = hr.fcluster(clustering, k, criterion="maxclust")
    clusters = {i: [] for i in range(min(clustering_inds), max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)

    ax = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    im = ax.pcolormesh(ordered_codep, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(codep.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(codep.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(labels[permutation], rotation=90, ha="center")
    ax.set_yticklabels(labels[permutation], va="center")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    flag = False
    if show_clusters is True:
        if linecolor is None:
            linecolor = 'fuchsia'
            flag = True
        elif linecolor is not None:
            flag = True

    if flag:
        for cluster_id, cluster in clusters.items():

            amin = permutation.index(cluster[0])
            xmin, xmax = amin, amin + len(cluster)
            ymin, ymax = amin, amin + len(cluster)

            for i in cluster:
                a = permutation.index(i)
                if a < amin:
                    xmin, xmax = a, a + len(cluster)
                    ymin, ymax = a, a + len(cluster)
                    amin = a

            ax.axvline(
                x=xmin, ymin=ymin / dim, ymax=(ymax) / dim, linewidth=4, color=linecolor
            )
            ax.axvline(
                x=xmax, ymin=ymin / dim, ymax=(ymax) / dim, linewidth=4, color=linecolor
            )
            ax.axhline(
                y=ymin, xmin=xmin / dim, xmax=(xmax) / dim, linewidth=4, color=linecolor
            )
            ax.axhline(
                y=ymax, xmin=xmin / dim, xmax=(xmax) / dim, linewidth=4, color=linecolor
            )

    axcolor = fig.add_axes([1.02, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)

    if dendrogram == True:

        ax1 = fig.add_axes([0.3, 0.71, 0.6, 0.2])

        if show_clusters is False:
            color_threshold = 0
        elif show_clusters is True:
            root, nodes = hr.to_tree(clustering, rd=True)
            nodes = [i.dist for i in nodes]
            nodes.sort()
            nodes = nodes[::-1][: k - 1]
            color_threshold = np.min(nodes)
            colors = af.color_list(k)
            hr.set_link_color_palette(colors)

        hr.dendrogram(
            clustering,
            color_threshold=color_threshold,
            above_threshold_color="grey",
            ax=ax1,
        )
        hr.set_link_color_palette(None)
        ax1.xaxis.set_major_locator(mticker.FixedLocator(np.arange(codep.shape[0])))
        ax1.set_xticklabels(labels[permutation], rotation=90, ha="center")

        if show_clusters is True:
            i = 0
            for coll in ax1.collections[:-1]:  # the last collection is the ungrouped level
                xmin, xmax = np.inf, -np.inf
                ymax = -np.inf
                for p in coll.get_paths():
                    (x0, _), (x1, y1) = p.get_extents().get_points()
                    xmin = min(xmin, x0)
                    xmax = max(xmax, x1)
                    ymax = max(ymax, y1)
                rec = plt.Rectangle(
                    (xmin - 4, 0),
                    xmax - xmin + 8,
                    ymax * 1.05,
                    facecolor=colors[i],  # coll.get_color()[0],
                    alpha=0.2,
                    edgecolor="none",
                )
                ax1.add_patch(rec)
                i += 1

        ax1.set_xticks([])
        ax1.set_yticks([])

        for i in {"right", "left", "top", "bottom"}:
            side = ax1.spines[i]
            side.set_visible(False)

        ax2 = fig.add_axes([0.09, 0.1, 0.2, 0.6])

        if show_clusters is True:
            hr.set_link_color_palette(colors)

        hr.dendrogram(
            clustering,
            color_threshold=color_threshold,
            above_threshold_color="grey",
            orientation="left",
            ax=ax2,
        )
        hr.set_link_color_palette(None)

        ax2.xaxis.set_major_locator(mticker.FixedLocator(np.arange(codep.shape[0])))
        ax2.set_xticklabels(labels[permutation], rotation=90, ha="center")

        if show_clusters is True:
            i = 0
            for coll in ax2.collections[:-1]:  # the last collection is the ungrouped level
                ymin, ymax = np.inf, -np.inf
                xmax = -np.inf
                for p in coll.get_paths():
                    (_, y0), (x1, y1) = p.get_extents().get_points()
                    ymin = min(ymin, y0)
                    ymax = max(ymax, y1)
                    xmax = max(xmax, x1)
                rec = plt.Rectangle(
                    (0, ymin - 4),
                    xmax * 1.05,
                    ymax - ymin + 8,
                    facecolor=colors[i],  # coll.get_color()[0],
                    alpha=0.2,
                    edgecolor="none",
                )
                ax2.add_patch(rec)
                i += 1

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        for i in {"right", "left", "top", "bottom"}:
            side = ax2.spines[i]
            side.set_visible(False)

    if title == "":
        title = (
            "Assets Clustermap ("
            + codependence.capitalize()
            + " & "
            + linkage
            + " " + method
            + " linkage)"
        )

    if dendrogram == True:
        ax1.set_title(title)
    elif dendrogram == False:
        ax.set_title(title)

    try:
        fig.tight_layout()
    except:
        pass

    return ax