function WordCloud(options) {
  var data = options.data//{"TF-IDF":{"심판":46.1713953676,"역사":44.4801395695,"나경원":28.9620729928,"국민":19.179685161,"나베":19.0329024594,"조국":18.9582603191,"수사":17.4323840035,"조사":16.9161373568,"구속":15.7236272529,"검찰":15.7084010269,"소리":11.0399480981,"한당":9.3730442357,"대표":7.8339981,"정치":7.1782452631,"국회":6.9881397617,"여권":6.9333334917,"패스트":6.8558099743,"자한":6.4458746213,"일본":6.3449887141,"비리":6.2237702509,"의원":6.1221410926,"트랙":5.9873222338,"야당":5.853190026,"자유":5.648867438,"여자":5.6384210094,"내년":5.6343938758,"응원":5.522962473,"처벌":5.3396501577,"자식":5.1916319027,"판사":5.189596363,"정권":5.1268830337,"정신":5.1027254565,"법대":5.0612283276,"총선":5.0585415666,"대한민국":5.0118696751,"한국":4.8808952807,"아들":4.8292535923,"나라":4.7329086436,"여당":4.6974239251,"무식":4.6170926486,"남불":4.5698979598,"경원":4.4300350278,"왜구":4.3967672688,"아베":4.3290103641,"수처":4.1508674457,"자유민주주의":4.1025393223,"출신":4.0807248669,"불법":3.9231017894,"재앙":3.8254486749,"개소리":3.8192082519,"걱정":3.8066052863,"기억":3.7314396998,"감옥":3.5714008775,"박근혜":3.4258349549,"자녀":3.3990324885,"인정":3.2814681035,"반대":3.2661614155,"쓰레기":3.1939615851,"화법":3.1438730377,"민주주의":3.141944082,"가족":3.1149734073,"인간":3.0896245838,"원칙":3.0804094674,"사형":3.0763338193,"선진":3.0681755162,"출석":3.0563979957,"독재":3.0278861552,"대통령":3.0200608055,"무능":2.9932758573,"정의":2.9783577512,"민주당":2.9061224658,"아가리":2.8811318468,"야권":2.835491029,"무도":2.8341291154,"타령":2.8192156875,"하늘":2.7267176021,"국당":2.69256144,"토착":2.686797008,"자신":2.6485191374,"주둥":2.6396656091,"보수":2.5986336944,"시간":2.5965532548,"사학":2.5387840055,"일가":2.4955241179,"감방":2.4907506433,"기록":2.4722446583,"좌파":2.448336356,"민주":2.4272075131,"당장":2.3742998148,"개혁":2.3717418482,"발목":2.3703921027,"얼굴":2.3401954697,"헛소리":2.3212050924,"극치":2.3154202771,"필요":2.2877907596,"날치기":2.2854638807,"이분":2.2801246071,"압수수색":2.2782213156,"민국":2.2736908161,"권력":2.2706082369}}

  var margin = {top: 70, right: 100, bottom: 0, left: 100},
           w = 1200 - margin.left - margin.right,
           h = 400 - margin.top - margin.bottom;

  // create the svg
  var svg = d3.select(options.container).append("svg")
              .attr('height', h + margin.top + margin.bottom)
              .attr('width', w + margin.left + margin.right)

  // set the ranges for the scales
  var xScale = d3.scaleLinear().range([10, 100]);

  var focus = svg.append('g')
                 .attr("transform", "translate(" + [w/2, h/2+margin.top] + ")")

  var colorMap = d3.schemeCategory10;
  //var colorMap = d3.scaleOrdinal(d3.schemeCategory10);

  // seeded random number generator
  var arng = new alea('hello.');

    var word_entries = d3.entries(data['TF-IDF']);
    xScale.domain(d3.extent(word_entries, function(d) {return d.value;}));

    makeCloud();

    function makeCloud() {
      d3.layout.cloud().size([w, h])
               .timeInterval(20)
               .words(word_entries)
               .fontSize(function(d) { return xScale(+d.value); })
               .text(function(d) { return d.key; })
               .font("Impact")
               .random(arng)
               .on("end", function(output) {
                 // sometimes the word cloud can't fit all the words- then redraw
                 // https://github.com/jasondavies/d3-cloud/issues/36
                 if (word_entries.length !== output.length) {
                   console.log("not all words included- recreating");
                   makeCloud();
                   return undefined;
                 } else { draw(output); }
               })
               .start();
    }

    d3.layout.cloud().stop();


  function draw(words) {
    focus.selectAll("text")
         .data(words)
         .enter().append("text")
         .style("font-size", function(d) { return xScale(d.value) + "px"; })
         .style("font-family", "Impact")
         .style("fill", function(d, i) { return colorMap[~~(arng() *10)]; })
         .attr("text-anchor", "middle")
         .attr("transform", function(d) {
           return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
         })
         .text(function(d) { return d.key; })
         .on('mouseover', handleMouseOver)
         .on('mouseout', handleMouseOut);
  }

  function handleMouseOver(d) {
    var group = focus.append('g')
                     .attr('id', 'story-titles');
     var base = d.y - d.size;

	var key = [d.key + ' : ' + String(data['TF-IDF'][d.key])]
    group.selectAll('text')
         .data(key)
         .enter().append('text')
         .attr('x', d.x)
         .attr('y', function(title, i) {
           return (base - i*14);
         })
         .attr('text-anchor', 'middle')
         .text(function(title) { return title; });

    var bbox = group.node().getBBox();
    var bboxPadding = 5;

    // place a white background to see text more clearly
    var rect = group.insert('rect', ':first-child')
                  .attr('x', bbox.x)
                  .attr('y', bbox.y)
                  .attr('width', bbox.width + bboxPadding)
                  .attr('height', bbox.height + bboxPadding)
                  .attr('rx', 10)
                  .attr('ry', 10)
                  .attr('class', 'label-background-strong');
  }

  function handleMouseOut(d) {
    d3.select('#story-titles').remove();
  }
}
