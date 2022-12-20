// phishing banner script

(function () {
    const template = `
<div id="phishing-banner">
    <div id="phishing-banner-content">
        <div>
            <div id="phishing-banner-title">Message from Axure</div>
            <div id="phishing-banner-text"> 
                <a id="phishing-banner-link" href="https://www.axure.com" target="_blank">Axure</a> strives to provide an easy and secure way to share ideas and prototypes. If you see content that may be deceptive, please <a id="phishing-banner-report">click here to report it</a>.
            </div>
        </div>
        <div id="phishing-banner-footer">
            <div id="phishing-banner-footer-tip">
                Why am I seeing this?
                <div id="tooltiptext">
                    <p>This message appears when a page contains a link that leaves Axure Cloud.</p>
                    <br/>
                    <p>You will only see this the first time you view the project. Setting an access code on the project prevents this message from appearing.</p>              
                </div>
            </div>
            <button id="phishing-banner-footer-button">
                OK
            </button>
    </div>
</div>
`;
    let $banner;
    let prototypeId;

    window.addEventListener("load", () => {
        init();
    });

    function init() {
        if (isCloud()) return;

        $axure.internal(function ($ax) {
            prototypeId = $ax.document.configuration.prototypeId;

            // check for is banner already viewed
            // and initially loaded page (skip for further)
            if (isBannerAlreadyViewed()
                || !isInitiallyLoadedPage()) return;

            // check for page has interactions with external link
            if (isInteractionMapHasExternalLinks($ax.pageData.page.interactionMap)) {
                showBanner();
                initInitiallyLoadedPageSafe(false);
                return;
            }

            // check for widgets has external links
            var scriptIds = $ax.getAllScriptIds();
            for (var i = 0; i < scriptIds.length; i++) {
                var diagramObject = $ax.getObjectFromScriptId(scriptIds[i]);

                if (isInlineFrameHasExternalLinks(diagramObject)
                    || isInteractionMapHasExternalLinks(diagramObject.interactionMap)) {
                    // has external links
                    showBanner();
                    initInitiallyLoadedPageSafe(false);
                    return;
                }
            }
            initInitiallyLoadedPageSafe(true);
        });
    }

    // look for external links in inlineFrame widget
    function isInlineFrameHasExternalLinks(dObj) {
        if (!dObj.type === "inlineFrame") {
            return false;
        }

        return isExternalTarget(dObj.target);
    }

    // look for external links in interactionMap
    function isInteractionMapHasExternalLinks(interactionMap) {
        if (!interactionMap) {
            return false;
        }

        let isExternalLinkFound = false;
        Object.values(interactionMap).every(value => {
            var cases = value.cases;
            if (cases) {
                cases.every(caseItem => {
                    var actions = caseItem.actions;
                    if (actions) {
                        actions.every(actionItem => {
                            if (actionItem.action === "linkWindow"
                                && isExternalTarget(actionItem.target)) {
                                isExternalLinkFound = true;
                            }
                            return !isExternalLinkFound; // break if external link found
                        });
                    }
                    return !isExternalLinkFound; // break if external link found
                });
            }
            return !isExternalLinkFound; // break if external link found
        });
        return isExternalLinkFound;
    }

    function isExternalTarget(target) {
        // if targetType is webUrl we can don't care about real URL
        // but skip if url is empty
        return target && target.targetType === "webUrl" && !isUrlLiteralEmpty(target.urlLiteral);
    }

    function isUrlLiteralEmpty(urlLiteral) {
        // empty if empty links ““ as well as links equal to "https://"
        return !urlLiteral || !urlLiteral.value || urlLiteral.value === "https://";
    }

    function showBanner() {
        var hostUrl = AXSHARE_HOST_SECURE_URL || window.AXSHARE_HOST_SECURE_URL;
        $.ajax( {
            type: 'GET',
            dataType: 'jsonp',
            url: hostUrl + '/prototype/showPhishingBanner?sk=' +prototypeId,
            success: function (response) {
                if(response.success) {
                    $banner = $(template).appendTo("body");
                    attachEvents($banner);
                    $banner.show();
                }
            }
        });
    }

    function attachEvents($banner) {
        $(document).on('click', $banner, function(e){ 
            e.stopPropagation();
        });
        $(document).on('click', '#phishing-banner-report', function(e){ 
            reportHandler(e); 
        });
        $(document).on('click', '#phishing-banner-footer-button', function(e){
            okHandler(e);
        });
        $(document).on('click', '#phishing-banner-link', function(e){ 
            e.stopPropagation();
        });
    }

    function getLocalStorageKey() {
        return `PB-${prototypeId}`;
    }

    function reportHandler(e) {
        e.stopPropagation();
        localStorage.setItem(getLocalStorageKey(), "report");
        const prototypeLink = document.location.href;
        reportPrototype(prototypeId, prototypeLink);

        if ($banner) {
            const $bannerText = $banner.find("#phishing-banner-text");
            $bannerText.text("Thank you for reporting this project. We will review it and take the appropriate action.");
        }
    }

    function okHandler(e) {
        e.stopPropagation();
        localStorage.setItem(getLocalStorageKey(), "ok");

        if ($banner) {
            $banner.hide();
        }
    }

    function isBannerAlreadyViewed() {
        if (localStorage.getItem(getLocalStorageKey())) {
            return true;
        }
        return false;
    }

    function isCloud() {
        const player = getPlayer();
        if (player && player.isCloud) {
            return true;
        }
        return false;
    }

    function initInitiallyLoadedPageSafe(isSafe) {
        const player = getPlayer();
        if (!player || player.isFirstLoadedPageSafe !== undefined) {
            return;
        }
        player.isFirstLoadedPageSafe = isSafe;
    }

    function isInitiallyLoadedPage() {
        const player = getPlayer();
        if (player) {
            return player.isFirstLoadedPageSafe === undefined;
        }
        return true;
    }

    function getPlayer() {
        const parentWindow = window.parent;
        if (parentWindow
            && parentWindow.$axure
            && parentWindow.$axure.player) {
            return parentWindow.$axure.player;
        }
        return undefined;
    }

    function reportPrototype(shortcut, link) {
        var hostUrl = AXSHARE_HOST_SECURE_URL || window.AXSHARE_HOST_SECURE_URL;

        $.ajax({
            type: "POST",
            url: hostUrl + '/prototype/reportprototype',
            data: { shortcut, link }
        });
    }
})();

